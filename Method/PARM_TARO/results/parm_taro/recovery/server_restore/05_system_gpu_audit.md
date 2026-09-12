# System and GPU audit

Status: **FAIL**.

PCI ID `10de:2b85` identifies an NVIDIA GeForce RTX 5090 (GB202); the NVIDIA 580.105.08 kernel module is loaded, but `/dev/nvidia*` is absent. `nvidia-smi` cannot communicate with the driver and a real CUDA tensor matmul fails with `No CUDA GPUs are available`. CUDA toolkit 11.5 being installed does not constitute a functioning runtime. No tiny-model smoke test was attempted because GPU compute already failed.

## uname -a

Exit code: `0`

```text
Linux iec-gv 5.15.0-190-generic #200-Ubuntu SMP Fri Aug 7 15:06:04 UTC 2026 x86_64 x86_64 x86_64 GNU/Linux
```

## /etc/os-release

Exit code: `0`

```text
PRETTY_NAME="Ubuntu 22.04.5 LTS"
NAME="Ubuntu"
VERSION_ID="22.04"
VERSION="22.04.5 LTS (Jammy Jellyfish)"
VERSION_CODENAME=jammy
ID=ubuntu
ID_LIKE=debian
HOME_URL="https://www.ubuntu.com/"
SUPPORT_URL="https://help.ubuntu.com/"
BUG_REPORT_URL="https://bugs.launchpad.net/ubuntu/"
PRIVACY_POLICY_URL="https://www.ubuntu.com/legal/terms-and-policies/privacy-policy"
UBUNTU_CODENAME=jammy
```

## nvidia-smi

Exit code: `0`

```text
Fri Sep 11 14:05:12 2026       
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 580.105.08             Driver Version: 580.105.08     CUDA Version: 13.0     |
+-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  NVIDIA GeForce RTX 5090        On  |   00000000:01:00.0 Off |                  N/A |
| 82%   71C    P1            529W /  600W |   29651MiB /  32607MiB |     99%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+

+-----------------------------------------------------------------------------------------+
| Processes:                                                                              |
|  GPU   GI   CI              PID   Type   Process name                        GPU Memory |
|        ID   ID                                                               Usage      |
|=========================================================================================|
|    0   N/A  N/A          104837      C   /opt/tljh/user/bin/python3              532MiB |
|    0   N/A  N/A         2623678      C   ...onda/envs/unimatch/bin/python      29102MiB |
+-----------------------------------------------------------------------------------------+
```

## nvidia-smi query

Exit code: `0`

```text
name, driver_version, memory.total [MiB], memory.free [MiB]
NVIDIA GeForce RTX 5090, 580.105.08, 32607 MiB, 2459 MiB
```

## nvcc --version

Exit code: `0`

```text
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2021 NVIDIA Corporation
Built on Thu_Nov_18_09:45:30_PST_2021
Cuda compilation tools, release 11.5, V11.5.119
Build cuda_11.5.r11.5/compiler.30672275_0
```

## python resolution

Exit code: `0`

```text
/home/jupyter-iec2024se10/.profile: line 29: ssh-rsa: command not found
/home/jupyter-iec2024se10/.profile: line 30: $'\E[H\E[2J\E[3J\E[H\E[2J\E[3J\E[H\E[2J\E[3J': command not found
bash: line 1: python: command not found
/usr/bin/python3
Python 3.10.12
```

## pip resolution

Exit code: `0`

```text
/home/jupyter-iec2024se10/.profile: line 29: ssh-rsa: command not found
/home/jupyter-iec2024se10/.profile: line 30: $'\E[H\E[2J\E[3J\E[H\E[2J\E[3J\E[H\E[2J\E[3J': command not found
/usr/bin/pip
pip 22.0.2 from /usr/lib/python3/dist-packages/pip (python 3.10)
```

## df -h

Exit code: `0`

```text
Filesystem                         Size  Used Avail Use% Mounted on
tmpfs                              6.3G  2.1M  6.3G   1% /run
/dev/mapper/ubuntu--vg-ubuntu--lv  2.7T  2.6T  9.5G 100% /
tmpfs                               32G  336K   32G   1% /dev/shm
tmpfs                              5.0M     0  5.0M   0% /run/lock
/dev/nvme0n1p2                     2.0G  434M  1.4G  24% /boot
/dev/nvme0n1p1                     1.1G  6.1M  1.1G   1% /boot/efi
tmpfs                              6.3G  8.0K  6.3G   1% /run/user/1004
tmpfs                              6.3G  8.0K  6.3G   1% /run/user/1015
tmpfs                              6.3G  8.0K  6.3G   1% /run/user/1019
tmpfs                              6.3G  8.0K  6.3G   1% /run/user/1000
tmpfs                              6.3G  8.0K  6.3G   1% /run/user/1012
tmpfs                              6.3G  8.0K  6.3G   1% /run/user/1020
```

## free -h

Exit code: `0`

```text
               total        used        free      shared  buff/cache   available
Mem:            62Gi        14Gi       1.3Gi       1.1Gi        46Gi        46Gi
Swap:          8.0Gi       1.7Gi       6.3Gi
```

## PCI GPU

Exit code: `0`

```text
01:00.0 VGA compatible controller [0300]: NVIDIA Corporation Device [10de:2b85] (rev a1)
	Kernel driver in use: nvidia
```

## Actual PyTorch CUDA compute

Exit code: `0`

```text
torch: 2.7.1+cu128
cuda runtime: 12.8
cuda available: True
gpu count: 1
CUDA_MATMUL_PASS 30.352628707885742
```
