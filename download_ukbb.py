import argparse
import getpass
import shlex
import sys
import tarfile
from pathlib import Path

try:
    import paramiko
except ImportError:
    sys.exit("paramiko is required: run `pip install paramiko` in your environment.")


SUBJECT_LIST_FILE = "subject_ids.txt"
DONE_FILE = ".downloaded_subjects.txt"


def parse_host(host: str) -> tuple[str, str | None]:
    """Splits 'user@hostname' into (hostname, user). User is None if absent."""
    if "@" in host:
        user, hostname = host.rsplit("@", 1)
        return hostname, user
    return host, None


def connect(host: str) -> paramiko.SSHClient:
    """
    Opens one SSH connection. Tries key/agent authentication first, then falls back to a
    single password prompt. The connection is reused for every batch so the password is
    only asked once.
    """
    hostname, username = parse_host(host)
    client = paramiko.SSHClient()
    client.load_system_host_keys()
    client.set_missing_host_key_policy(paramiko.WarningPolicy())

    print(f"Connecting to {host}...")
    try:
        client.connect(hostname, username=username, look_for_keys=True, allow_agent=True, timeout=30)
    except (paramiko.AuthenticationException, paramiko.SSHException):
        password = getpass.getpass(f"Password for {host}: ")
        client.connect(hostname, username=username, password=password,
                       look_for_keys=False, allow_agent=False, timeout=30)

    client.get_transport().set_keepalive(30)
    return client


def run_remote(client: paramiko.SSHClient, cmd: str) -> str:
    """Runs a short remote command and returns its stdout. Raises on non-zero exit."""
    _, stdout, stderr = client.exec_command(cmd)
    out = stdout.read().decode(errors="replace")
    status = stdout.channel.recv_exit_status()
    if status != 0:
        err = stderr.read().decode(errors="replace").strip()
        raise RuntimeError(f"Remote command failed (exit {status}): {cmd}\n{err}")
    return out


def sort_key(subject_id: str):
    """Numeric IDs sort lowest to highest; anything non-numeric goes after them alphabetically."""
    return (0, int(subject_id)) if subject_id.isdigit() else (1, subject_id)


def fetch_subject_ids(client: paramiko.SSHClient, remote_base_dir: str, cache_file: Path, refresh: bool) -> list[str]:
    """
    Returns the sorted list of subject IDs in the remote dataset directory.
    The list is cached locally so the (slow) remote listing only happens once.
    """
    if cache_file.exists() and not refresh:
        ids = [line.strip() for line in cache_file.read_text().splitlines() if line.strip()]
        print(f"Using cached subject list ({len(ids)} subjects) from {cache_file}")
        return ids

    print("Listing remote subjects (this can take a while on a large directory)...")
    # Plain `ls -1` avoids stat-ing every entry, which is what makes `find -type d` slow on NFS.
    out = run_remote(client, f"cd {shlex.quote(remote_base_dir)} && ls -1")
    ids = sorted((name.strip() for name in out.splitlines() if name.strip() and not name.startswith(".")),
                 key=sort_key)
    cache_file.write_text("\n".join(ids) + "\n")
    print(f"Found {len(ids)} subjects; list cached to {cache_file}")
    return ids


def load_done(done_file: Path) -> set[str]:
    if not done_file.exists():
        return set()
    return {line.strip() for line in done_file.read_text().splitlines() if line.strip()}


def download_batch(
        client: paramiko.SSHClient,
        remote_base_dir: str,
        local_base_dir: Path,
        subject_ids: list[str],
) -> int:
    """Streams one tar archive containing the given subjects and extracts it. Returns the file count."""
    quoted = " ".join(shlex.quote(s) for s in subject_ids)
    remote_cmd = f"cd {shlex.quote(remote_base_dir)} && tar -cf - -h {quoted}"

    channel = client.get_transport().open_session()
    channel.exec_command(remote_cmd)
    stream = channel.makefile("rb")

    file_count = 0
    current_subject = None
    subject_files = 0
    try:
        with tarfile.open(fileobj=stream, mode="r|") as tar:
            for member in tar:
                subject = Path(member.name).parts[0] if Path(member.name).parts else member.name
                if subject != current_subject:
                    if current_subject is not None:
                        print(f"    {current_subject}: {subject_files} files")
                    current_subject, subject_files = subject, 0

                if sys.version_info >= (3, 11, 4):
                    tar.extract(member, path=local_base_dir, filter="data")
                else:
                    tar.extract(member, path=local_base_dir)

                if member.isfile() or member.issym():
                    subject_files += 1
                    file_count += 1
        if current_subject is not None:
            print(f"    {current_subject}: {subject_files} files")
    except tarfile.ReadError as e:
        raise RuntimeError(f"Failed to read tar stream: {e}") from e

    status = channel.recv_exit_status()
    err = channel.makefile_stderr("rb").read().decode(errors="replace").strip()
    channel.close()
    if status != 0:
        raise RuntimeError(f"Remote tar exited with code {status}:\n{err}")
    if err:
        print(f"    (remote warnings)\n{err}")
    return file_count


def download_ukbb_in_batches(
        remote_host: str,
        remote_base_dir: str,
        local_base_dir: Path,
        n_subjects: int,
        batch_size: int,
        refresh_list: bool,
) -> None:
    local_base_dir.mkdir(parents=True, exist_ok=True)
    cache_file = local_base_dir / SUBJECT_LIST_FILE
    done_file = local_base_dir / DONE_FILE

    client = connect(remote_host)
    try:
        all_ids = fetch_subject_ids(client, remote_base_dir, cache_file, refresh_list)
        wanted = all_ids if n_subjects < 0 else all_ids[:n_subjects]

        done = load_done(done_file)
        todo = [s for s in wanted if s not in done]
        print(f"{len(wanted)} subjects requested, {len(wanted) - len(todo)} already downloaded, "
              f"{len(todo)} to go in batches of {batch_size}.\n")

        batches = [todo[i:i + batch_size] for i in range(0, len(todo), batch_size)]
        total_files = 0
        for i, batch in enumerate(batches, start=1):
            print(f"Batch {i}/{len(batches)}: subjects {batch[0]} .. {batch[-1]}")
            total_files += download_batch(client, remote_base_dir, local_base_dir, batch)
            with done_file.open("a") as f:
                f.write("\n".join(batch) + "\n")

        print(f"\nDone. Downloaded {total_files} files for {len(todo)} new subjects "
              f"({len(wanted)} total available locally).")
    finally:
        client.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download subjects from the UKBB cardiac segmentations dataset in sorted batches "
                    "over a single SSH connection.")
    parser.add_argument(
        "--host",
        type=str,
        default="stol@131.159.110.9",
        help="SSH host (e.g., user@server)"
    )
    parser.add_argument(
        "--remote-dir",
        type=str,
        default="/vol/miltank/projects/ukbb/data/cardiac/cardiac_segmentations/projects/ugurlu_segmentations/data",
        help="Remote dataset directory"
    )
    parser.add_argument(
        "--local-dir",
        type=Path,
        default=Path(r"C:\Users\Nil\Documents\data\ukbb"),
        help="Local destination directory"
    )
    parser.add_argument(
        "--n-subjects",
        type=int,
        default=25,
        help="Number of subjects to download, taken from the lowest ID upwards. Use -1 for all."
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=10,
        help="Subjects per tar stream"
    )
    parser.add_argument(
        "--refresh-list",
        action="store_true",
        help="Re-fetch the remote subject list instead of using the cached one"
    )

    args = parser.parse_args()
    download_ukbb_in_batches(
        remote_host=args.host,
        remote_base_dir=args.remote_dir,
        local_base_dir=args.local_dir,
        n_subjects=args.n_subjects,
        batch_size=args.batch_size,
        refresh_list=args.refresh_list,
    )
