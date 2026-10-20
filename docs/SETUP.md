# Setup and local connectivity

Status: researched instructions, **unverified on the target Windows RTX 4050 and M4 Mac**. Local [M1](M1.md) diagnostics, [M2](M2.md) transport probe and [M3](M3.md) real serial training launches now exist. Commands below are independent diagnostics or package/setup commands. Sources and compatibility qualifications are in [platform evidence](research/PLATFORM.md), accessed 2026-10-07. Budget 6 GB dedicated NVIDIA memory and 16 GB shared Mac memory separately; see [FEASIBILITY](FEASIBILITY.md).

## 1. Prepare matching native Python environments

Use CPython **3.13, standard GIL build**, PyTorch **2.11.0**, Windows CUDA **12.8** wheel and macOS ARM64 wheel. Exact wheel names were verified in the official indexes; installation and operation still require M0 checks. This is a pinned compatibility candidate, not the latest-release claim. Keep the same source checkout on both machines and a separate `.venv` on each. Never copy the virtual environment across operating systems.

Install a current Python 3.13 patch from [python.org](https://www.python.org/downloads/): Windows 64-bit installer with launcher, macOS universal2 installer run natively. An existing native 3.13 installation is sufficient. Do not select a free-threaded build. [Python macOS instructions](https://docs.python.org/3.13/using/mac.html) explain native/universal installers; [venv documentation](https://docs.python.org/3.13/library/venv.html) describes isolation. Get the project folder onto both machines by your usual file-transfer method; this guide does not assume a remote Git URL.

**Machine A; ordinary PowerShell; working directory: project root.** The following assumes the folder is in your user Documents directory; adjust only the first path if needed.

```powershell
Set-Location "$env:USERPROFILE\Documents\hive-train"
Get-CimInstance Win32_OperatingSystem | Select-Object Caption,Version,OSArchitecture
py -3.13 -c "import sys,platform; print(sys.version); print(platform.machine()); print(sys.executable)"
nvidia-smi
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128
.\.venv\Scripts\python.exe -m pip check
```

Expected: 64-bit Windows/Python, RTX 4050 listed, compatible NVIDIA driver, successful package install, and `No broken requirements found`. Prefer a supported Windows 11 installation; record the actual version before selecting updates. `py` missing means install/repair the Python launcher or use the verified absolute Python 3.13 executable. `No matching distribution` means check Python minor version, 64-bit architecture, index/network access, and pip version. For DLL load errors inspect the Windows FAQ and required Microsoft runtime before changing packages. Calling the venv executable directly avoids PowerShell activation-policy changes.

The compatible NVIDIA driver is a system prerequisite. For this cu128 candidate, use a suitable laptop driver at least **570.65**, then test; see [driver qualifications](research/PLATFORM.md). A working newer driver need not be replaced. CUDA runtime libraries come with the selected binary distribution/dependencies. A standalone CUDA Toolkit and `nvcc` are unnecessary for the planned eager PyTorch operations. The `CUDA Version` line in `nvidia-smi` is driver capability, not the installed wheel's runtime version.

**Machine B; Terminal using zsh; working directory: project root.**

```zsh
cd "$HOME/Documents/hive-train"
sw_vers
uname -m
python3.13 -c 'import sys,platform; print(sys.version); print(platform.machine()); print(sys.executable)'
python3.13 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cpu
.venv/bin/python -m pip check
```

Expected: `arm64`, macOS 14+ (M4's actual installed version must be recorded), successful install and dependency check. The index named `cpu` also distributes the ARM64 macOS wheel with MPS. `x86_64` indicates an Intel interpreter or Rosetta Terminal: use native Terminal/Python and recreate only this project's venv after preserving anything needed from it. A wheel tag alone does not establish MPS availability. No CUDA installation applies to the Mac. If a tool specifically requires Apple command-line tools, check `xcode-select -p` and then use `xcode-select --install`; full Xcode is not an M0 requirement.

## 2. Verify accelerator forward, backward, and update

**A; PowerShell; project root:**

```powershell
.\.venv\Scripts\python.exe -c "import torch; print(torch.__version__, torch.version.cuda); assert torch.cuda.is_available(); torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False; print(torch.cuda.get_device_name(0)); print(torch.cuda.get_device_properties(0)); x=torch.randn(16,32,device='cuda',requires_grad=True); m=torch.nn.Sequential(torch.nn.Linear(32,32),torch.nn.GELU(),torch.nn.LayerNorm(32)).to('cuda'); o=torch.optim.AdamW(m.parameters(),lr=0.001,foreach=False,fused=False); before=m[0].weight.detach().clone(); loss=m(x).square().mean(); loss.backward(); assert x.grad is not None and torch.isfinite(x.grad).all(); assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters()); o.step(); torch.cuda.synchronize(); assert not torch.equal(before,m[0].weight); print('PASS CUDA',float(loss.detach().cpu()),torch.cuda.max_memory_allocated())"
```

**B; zsh; project root:** disable fallback before importing torch.

```zsh
PYTORCH_ENABLE_MPS_FALLBACK=0 .venv/bin/python -c 'import torch; print(torch.__version__,torch.backends.mps.is_built(),torch.backends.mps.is_available()); assert torch.backends.mps.is_available(); x=torch.randn(16,32,device="mps",requires_grad=True); m=torch.nn.Sequential(torch.nn.Linear(32,32),torch.nn.GELU(),torch.nn.LayerNorm(32)).to("mps"); o=torch.optim.AdamW(m.parameters(),lr=0.001,foreach=False,fused=False); before=m[0].weight.detach().clone(); loss=m(x).square().mean(); loss.backward(); assert x.grad is not None and torch.isfinite(x.grad).all(); assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters()); o.step(); torch.mps.synchronize(); assert not torch.equal(before,m[0].weight); print("PASS MPS",float(loss.detach().cpu()),torch.mps.current_allocated_memory(),torch.mps.driver_allocated_memory())'
```

Expected: a `PASS` label, finite loss, nonzero allocation and changed weight; values are intentionally not prescribed. CUDA unavailable: confirm cu128 wheel, driver/device status, and reboot if a driver update required it. MPS unavailable: check native Python, OS and MPS build flags. Unsupported MPS operation: retain the traceback and change/verify the operation in M1; do not enable fallback and call the result GPU performance. These small tests do not establish decoder coverage, numerical equivalence, or training capacity. M1 adds embedding, causal attention, softmax, cross-entropy, all parameter gradients and update comparisons.

Record exact package state (each command writes only an environment report, not application configuration):

```powershell
# A; PowerShell; project root
.\.venv\Scripts\python.exe -m pip freeze > environment-windows.txt
```

```zsh
# B; zsh; project root
.venv/bin/python -m pip freeze > environment-macos.txt
```

Expected: platform-specific reports including torch 2.11.0 (Windows `+cu128` suffix is expected). Also record OS, driver, Python patch, power mode and whether either machine swaps memory. Do not use one OS's freeze report to install the other OS's torch wheel.

## 3. Discover the local network

Connect both machines to the same authorized local network. For college Wi-Fi, being on the same SSID does not prove peer reachability. Prefer a password-protected private hotspot for prototype plaintext tensor traffic. Download dependencies/data first; training itself must not require internet service.

**A; PowerShell; any directory:**

```powershell
Get-NetIPConfiguration
Get-NetConnectionProfile
ipconfig
```

Expected: an active Wi-Fi or hotspot virtual adapter with IPv4 address and subnet. Select its local address, not VPN, WSL, Bluetooth or disconnected adapter addresses. `169.254.*` usually means address assignment failed. Record this as `WINDOWS_IP`.

**B; zsh; any directory:**

```zsh
networksetup -listallhardwareports
route -n get default
ipconfig getifaddr en0
```

Expected: hardware-port listing identifies the Wi-Fi device. Replace `en0` if that listing names another interface; it is not guaranteed to be en0. Record the Wi-Fi IPv4 address as `MAC_IP`. A VPN default route can mislead you; use the actual Wi-Fi interface and local routing policy.

For a Windows-hosted hotspot use Settings → Network & internet → Mobile hotspot → share over Wi-Fi, select a password and suitable band, then connect the Mac. Re-run address discovery because the Windows hotspot adapter address differs from the upstream Wi-Fi address. If unavailable under hardware/college policy, try a permitted phone hotspot with both laptops attached. Some phones isolate clients or require an active uplink; verify with the test below. No router port forwarding is needed. [Microsoft instructions](https://support.microsoft.com/en-us/windows/experience/connectivity-networking/use-your-windows-device-as-a-mobile-hotspot).

## 4. Allow only the required Windows listener

**A; administrator PowerShell; project root:** first determine the active profile from step 3. On a trusted personal hotspot use `Private`; if the actual interface is `Public`, keep it Public and substitute that profile below with the same narrow address/program scope. Do not reclassify college Wi-Fi as trusted. Enter the recorded Mac address at the prompt.

```powershell
Set-Location "$env:USERPROFILE\Documents\hive-train"
$macPeer = Read-Host 'Mac Wi-Fi IPv4 address'
$windowsLocal = Read-Host 'Windows Wi-Fi or hotspot local IPv4 address'
$pythonExe = (Resolve-Path .\.venv\Scripts\python.exe).Path
New-NetFirewallRule -Name 'HiveTrain-29550' -DisplayName 'HiveTrain TCP 29550 from Mac' -Direction Inbound -Action Allow -Protocol TCP -LocalPort 29550 -LocalAddress $windowsLocal -Program $pythonExe -RemoteAddress $macPeer -Profile Private
Get-NetFirewallRule -Name 'HiveTrain-29550'
```

Expected: an enabled, inbound allow rule for one Python executable, peer and port. Access denied: use an administrator terminal or ask the machine/network administrator; do not disable the firewall. Existing rule: inspect it rather than repeatedly creating duplicates. DHCP/IP changes require updating the peer address. Windows venv launchers can involve a base Python executable; if diagnostics show the process's listener belongs to that executable, scope a replacement rule to its verified path, not all Python programs. No Mac inbound rule is needed: the Mac initiates its connection. See [Microsoft rule parameters](https://learn.microsoft.com/en-us/powershell/module/netsecurity/new-netfirewallrule).

After testing, remove only the project rule if no longer needed (A, administrator PowerShell, any directory):

```powershell
Remove-NetFirewallRule -Name 'HiveTrain-29550'
```

Expected: this one allow rule is removed; recreate it before a later run. Do not remove unrelated rules.

## 5. Test TCP locally, then across Wi-Fi

This diagnostic uses only synthetic bytes and is not the application protocol or an authentication test. Restrict it to the trusted network and close it afterward. It bounds payloads and reads complete frames, because one `recv` need not return a complete message. The diagnostic binds one selected address per test; the future runtime binds `0.0.0.0:29550` to accept loopback and LAN workers together, with the scoped external firewall rule above.

**A; ordinary PowerShell; project root:** enter `127.0.0.1` first to test loopback, then rerun with `WINDOWS_IP` for the Mac test. The server handles one connection and exits when the client disconnects. Keep this terminal open.

```powershell
$env:HIVE_BIND = Read-Host 'Bind IPv4 (127.0.0.1 for loopback, then Windows local IP)'
@'
import os, socket, struct
def read_exact(c, n):
    b = bytearray(n)
    v = memoryview(b)
    while v:
        k = c.recv_into(v)
        if not k: raise EOFError("peer closed")
        v = v[k:]
    return b
with socket.socket() as s:
    s.settimeout(120)
    s.bind((os.environ["HIVE_BIND"], 29550))
    s.listen(1)
    print("LISTENING", s.getsockname(), flush=True)
    c, addr = s.accept()
    with c:
        c.settimeout(30)
        c.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        print("CONNECTED", addr, flush=True)
        try:
            while True:
                op, n = struct.unpack("!cI", read_exact(c, 5))
                if n > 64*1024*1024: raise ValueError("oversize")
                if op == b"U":
                    read_exact(c, n)
                    c.sendall(b"K")
                elif op == b"D":
                    c.sendall(bytes(n))
                elif op == b"E":
                    c.sendall(read_exact(c, n))
                else: raise ValueError("unknown operation")
        except EOFError:
            print("CLIENT CLOSED")
'@ | .\.venv\Scripts\python.exe -
```

Expected: `LISTENING`, then `CONNECTED` when a client arrives. For loopback, run this in a **second A PowerShell terminal, project root**, before restarting the server for the Mac:

```powershell
.\.venv\Scripts\python.exe -c "import socket,struct; s=socket.create_connection(('127.0.0.1',29550),5); s.sendall(struct.pack('!cI',b'E',4)+b'hive'); f=s.makefile('rb'); assert f.read(4)==b'hive'; print('PASS loopback'); f.close(); s.close()"
```

Expected: `PASS loopback`. Now restart the server bound to `WINDOWS_IP`, and run the client below on **B; zsh; project root**. The prompt takes `WINDOWS_IP`; do not use a public address.

```zsh
read 'HIVE_HOST?Windows local IPv4: '
export HIVE_HOST
.venv/bin/python - <<'PY'
import os, socket, statistics, struct, time
def read_exact(s, n):
    b = bytearray(n)
    v = memoryview(b)
    while v:
        k = s.recv_into(v)
        if not k: raise EOFError("peer closed")
        v = v[k:]
    return b
def report(label, samples):
    a = sorted(samples)
    print(label, "p50_ms", statistics.median(a)*1000,
          "p95_ms", a[max(0, int(0.95*len(a))-1)]*1000,
          "max_ms", max(a)*1000)
with socket.create_connection((os.environ["HIVE_HOST"], 29550), 10) as s:
    s.settimeout(30)
    s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    latency = []
    for i in range(110):
        t = time.perf_counter()
        s.sendall(struct.pack("!cI", b"E", 4) + b"hive")
        assert read_exact(s, 4) == b"hive"
        if i >= 10: latency.append(time.perf_counter()-t)
    print("PASS cross-machine TCP")
    report("4-byte application RTT", latency)
    for n in (1024, 1024**2, 4*1024**2, 16*1024**2, 64*1024**2):
        payload = bytes(n)
        for op, label in ((b"U", "Mac->Windows"), (b"D", "Windows->Mac")):
            samples = []
            for i in range(12):
                t = time.perf_counter()
                s.sendall(struct.pack("!cI", op, n))
                if op == b"U":
                    s.sendall(payload)
                    assert read_exact(s, 1) == b"K"
                else:
                    received = read_exact(s, n)
                elapsed = time.perf_counter()-t
                if i >= 2: samples.append(elapsed)
            report(f"{label} bytes={n}", samples)
            print("median_effective_MiB_s", n/1024**2/statistics.median(samples))
PY
```

Expected: TCP pass, latency distribution, and separate directional effective throughputs. No expected numerical results are supplied. Transfer timers include request/acknowledgement and receiver allocation; they are application-level directional costs, not synchronized one-way latency. Zero bytes are suitable for this uncompressed raw TCP capacity probe; M2 must additionally test random binary tensors and exact payload equality. Ten throughput observations are preliminary: repeat sessions at different times, stationary placement, AC power, then use the longer protocol-realistic profiling in [VALIDATION](VALIDATION.md). Internet download speed is irrelevant to this test. On a very slow network a 64 MiB frame can legitimately exceed the diagnostic 30-second timeout: start with smaller sizes or consciously increase that diagnostic timeout and record it.

Troubleshooting: connection refused means wrong IP or no listener; timeout can mean firewall, routing, isolation, or a stalled machine. While the server is listening, A can inspect `Get-NetTCPConnection -LocalPort 29550 -State Listen` (PowerShell, any directory). Address-in-use means another process is already listening; stop the diagnostic/application you started, not an unrelated process. If loopback succeeds but cross-machine TCP fails, verify bind IP, peer IP, firewall profile/program and VPN routes. Run the same test on a private hotspot: success there plus persistent college failure is evidence of college isolation/filtering; ask network administrators rather than bypassing policy. Ping failure alone does not prove isolation.

## 6. Separate accelerator staging costs and retain M0 evidence

Before GPU communication claims, measure CPU→device and device→CPU separately for FP32 tensors of 1, 4, 16 and 64 MiB (and each actual boundary shape). In M1/M2's proposed profiler, allocate a contiguous CPU tensor and device tensor once; synchronize the device; time a blocking `.to(device)`; synchronize again; then independently time `.detach().to('cpu').contiguous()` with synchronization. Repeat after warmup and report median/p95, along with serialization/host-copy timing. Count any allocation in a path that allocates; also measure buffer-reuse copies separately. Use `torch.cuda.synchronize()` on A and `torch.mps.synchronize()` on B. Synchronize before and after GPU timing: dispatch latency alone is not execution time. MPS unified memory does not justify assuming staging and synchronization are free.

M0 output is an environment/network report with exact commands, hardware/OS/Python/torch/driver facts, pass/fail traces, selected local IPs and profile, and measured local RTT/directional throughput. Record unavailable checks as pending; do not substitute invented outputs. [M1](M1.md) provides reusable local model/gradient diagnostics; [M2](M2.md) now provides an authenticated transport probe with exact payload checks and JSON reports. A reusable accelerator-copy profiler remains future work. Milestones: [IMPLEMENTATION_PLAN](IMPLEMENTATION_PLAN.md). GPU fallback and quantitative equivalence rules: [VALIDATION](VALIDATION.md).
