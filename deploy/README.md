# Keeping a Diffuse host running

By default `diffuse host` runs in the foreground: you see the logs, and Ctrl+C
stops it. To contribute to the network continuously, keep it running in the
background with one of these.

## Simple: nohup

Runs detached, survives closing the terminal:

```bash
nohup diffuse host --model Qwen/Qwen2.5-0.5B-Instruct > ~/.diffuse/host.log 2>&1 &
```

- Check on it: `tail -f ~/.diffuse/host.log`
- Stop it: `pkill diffuse`

This does not restart after a machine reboot.

## Permanent: systemd (Linux)

For a node that also restarts on boot, use the provided service file.

Assuming the binary is at `~/.local/bin/diffuse` and the worker at
`~/.diffuse/worker`, create a service (edit paths if yours differ):

```bash
mkdir -p ~/.config/systemd/user
cat > ~/.config/systemd/user/diffuse-host.service << 'UNIT'
[Unit]
Description=Diffuse host node
After=network.target

[Service]
Type=simple
Environment=DIFFUSE_WORKER_DIR=%h/.diffuse/worker
ExecStart=%h/.local/bin/diffuse host --model Qwen/Qwen2.5-0.5B-Instruct
Restart=always
RestartSec=10

[Install]
WantedBy=default.target
UNIT

systemctl --user daemon-reload
systemctl --user enable --now diffuse-host
loginctl enable-linger "$USER"
```

- Status: `systemctl --user status diffuse-host`
- Logs: `journalctl --user -u diffuse-host -f`
- Stop: `systemctl --user stop diffuse-host`

The `enable-linger` line lets the service run even when you are not logged in.

A system-wide template is also provided in `diffuse-host.service` for
`/etc/systemd/system/` if you prefer a root-managed service.
