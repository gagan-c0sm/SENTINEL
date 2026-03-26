# SENTINEL — Database Transfer Guide

> **Goal**: Effectively transfer the 5.6 GB TimescaleDB instance from one Docker host to another.
>
> [!NOTE]
> **What is a Docker Image?**  
> An image is like a "software blueprint" (e.g., Postgres + TimescaleDB pre-installed). You **do NOT** need to copy the image itself; the new computer will automatically download it from the internet when you run `docker-compose up`. You only need to move your **Data** and **Configuration** files.

---

## Method 1: Database Export/Import (Recommended)
This is the most robust method for moving between different operating systems or Docker versions.

### 1. On the SOURCE Computer
Run this command to create a compressed backup of the entire database. This uses PostgreSQL's internal compression format (`-Fc`):

```powershell
docker exec sentinel-db pg_dump -U sentinel -d sentinel -Fc -f /tmp/sentinel_backup.dump
docker cp sentinel-db:/tmp/sentinel_backup.dump sentinel_backup.dump
docker exec sentinel-db rm /tmp/sentinel_backup.dump
```
> [!NOTE]
> **Why `docker cp`?** PowerShell's `>` redirection corrupts binary data by encoding it as UTF-16. This method ensures the file remains a valid binary backup.
>
> **About pg_dump Warnings**: You may see "circular foreign-key constraints" warnings for hypertables and chunks. **These are normal for TimescaleDB** and can be safely ignored; the `--disable-triggers` flag in the restore command handles them.

### 2. Move the File
Transfer `sentinel_backup.dump` to the destination computer.

### 3. On the DESTINATION Computer
1. Start the project’s Docker container (ensure `docker-compose.yml` is present):
   ```powershell
   docker-compose up -d
   ```
2. Restore the data using the `pg_restore` command:
   ```powershell
   # Restore into the new container (the --disable-triggers flag handles the FK warnings)
   docker exec -i sentinel-db pg_restore -U sentinel -d sentinel --disable-triggers < sentinel_backup.dump
   ```

---

## Method 2: Volume Mirroring (Advanced)
If you want to skip the dump/restore process and just move the raw data files.

### 1. Stop the Source Container
```powershell
docker-compose down
```

### 2. Locate the Task Volume
Find where Docker stores the `sentinel_db_data` volume on your host. By default, it's usually inside:
- **Linux**: `/var/lib/docker/volumes/`
- **Windows (Docker Desktop)**: Inside the WSL2 `docker-desktop-data` VHDX.

### 3. Copy & Replace
1. Zip the entire volume directory.
2. Transfer it to the destination.
3. Start the destination container once to initialize the volume, then stop it.
4. Replace the contents of the destination volume with your source zip.

---

## Technical Note on TimescaleDB
Since you are using TimescaleDB, the destination computer **MUST** use the same Docker image (`timescale/timescaledb:latest-pg16`) to ensure extension compatibility. Standard `pg_dump` is safe for hypertables as long as the extension is pre-installed (which it is by the `init.sql` script).

> [!IMPORTANT]
> **Don't Forget the Environment**: Remember to also move your `.env` file and the `d:\Projects\SENTINEL` source code. The database alone is not enough; the virtual environment and file paths must also be configured on the new machine.
