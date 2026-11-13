"""PrivAI - Day 6: Network monitor (proof that nothing leaves your laptop).

Watches Python / Streamlit / Ollama and records every connection that goes
anywhere other than your own computer (loopback).

Run in a SECOND terminal while you use the app:
    python net_monitor.py 120        # watch for 120 seconds
"""
import csv
import ipaddress
import sys
import time

import psutil

WATCH = ("python", "streamlit", "ollama", "llama")
LOG = "network_log.csv"


def own_addresses():
    """All IP addresses that belong to THIS computer (Wi-Fi, LAN, loopback)."""
    ips = set()
    for addrs in psutil.net_if_addrs().values():
        for a in addrs:
            ips.add(a.address.split("%")[0])
    return ips


def classify(ip, own):
    """'own' = this computer, 'lan' = another device nearby, 'internet' = outside."""
    if ip in own:
        return "own"
    try:
        addr = ipaddress.ip_address(ip)
        if getattr(addr, "ipv4_mapped", None):
            addr = addr.ipv4_mapped
            if str(addr) in own:
                return "own"
        if addr.is_loopback:
            return "own"
        if addr.is_private or addr.is_link_local:
            return "lan"
        return "internet"
    except ValueError:
        return "own"


def watched_pids():
    pids = {}
    for p in psutil.process_iter(["pid", "name"]):
        name = (p.info["name"] or "").lower()
        if any(w in name for w in WATCH):
            pids[p.info["pid"]] = name
    return pids


def main(seconds):
    seen, rows = set(), []
    own = own_addresses()
    end = time.time() + seconds
    print(f"Watching for {seconds}s... use the app now (chat, RAG, agent).")
    while time.time() < end:
        pids = watched_pids()
        try:
            conns = psutil.net_connections(kind="inet")
        except psutil.AccessDenied:
            sys.exit("Access denied: run this terminal as Administrator.")
        for c in conns:
            if c.pid in pids and c.raddr:
                kind = classify(c.raddr.ip, own)
                if kind == "own":
                    continue                      # talking to itself = fine
                key = (c.pid, c.raddr.ip)         # one line per remote address
                if key not in seen:
                    seen.add(key)
                    rows.append([time.strftime("%H:%M:%S"), kind.upper(),
                                 pids[c.pid], c.pid, c.raddr.ip, c.status])
                    print(f"{kind.upper():8} ->", rows[-1][2:])
        time.sleep(1)

    with open(LOG, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["time", "kind", "process", "pid", "remote_ip", "status"])
        w.writerows(rows)
    internet = [r for r in rows if r[1] == "INTERNET"]
    lan = [r for r in rows if r[1] == "LAN"]
    print(f"\nChecked {seconds}s.  Internet connections: {len(internet)}   "
          f"Other devices on your network (LAN): {len(lan)}")
    if internet:
        print("RESULT: CHECK - something talked to the internet (see list above)")
    elif lan:
        print("RESULT: PASS for internet. But another device on your network "
              "connected - make sure Streamlit only listens on localhost.")
    else:
        print("RESULT: PASS - nothing left this machine")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 60)