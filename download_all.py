import os
import sys
import struct
import zlib
import urllib.request
import time

def main():
    with open('zip_tail.bin', 'rb') as f:
        data = f.read()

    entries = {}
    offset = 0
    while offset < len(data) - 46:
        if data[offset:offset+4] != b'PK\x01\x02':
            break
        header = data[offset:offset+46]
        sig, ver_made, ver_need, flags, method, time_val, date, crc, comp_sz, uncomp_sz, fname_len, extra_len, comment_len, disk_num, int_attr, ext_attr, local_hdr_off = struct.unpack('<4sHHHHHHIIIHHHHHII', header)
        fname = data[offset+46:offset+46+fname_len].decode('utf-8', errors='replace')
        extra = data[offset+46+fname_len:offset+46+fname_len+extra_len]
        e_off = 0
        actual_uncomp_sz, actual_comp_sz, actual_local_hdr_off = uncomp_sz, comp_sz, local_hdr_off
        while e_off < len(extra) - 4:
            tag, sz = struct.unpack('<HH', extra[e_off:e_off+4])
            e_off += 4
            if tag == 0x0001:
                idx = e_off
                if actual_uncomp_sz == 0xFFFFFFFF:
                    actual_uncomp_sz, = struct.unpack('<Q', extra[idx:idx+8]); idx += 8
                if actual_comp_sz == 0xFFFFFFFF:
                    actual_comp_sz, = struct.unpack('<Q', extra[idx:idx+8]); idx += 8
                if actual_local_hdr_off == 0xFFFFFFFF:
                    actual_local_hdr_off, = struct.unpack('<Q', extra[idx:idx+8]); idx += 8
            e_off += sz
        entries[fname] = {'method': method, 'comp_size': actual_comp_sz, 'uncomp_size': actual_uncomp_sz, 'local_hdr_off': actual_local_hdr_off}
        offset += 46 + fname_len + extra_len + comment_len

    feature_tasks = set()
    for l in entries:
        parts = l.split('/')
        if len(parts) >= 6 and parts[2] == 'Lab' and parts[3] == 'Features' and parts[5].endswith('30_sec.pickle'):
            feature_tasks.add((parts[1], parts[4]))

    targets = []
    for p, _ in feature_tasks:
        l_path = f'EPIStress/{p}/Lab/Task_Labels.csv'
        if l_path in entries:
            targets.append(l_path)
    for l in entries:
        if 'Lab/Features' in l and l.endswith('30_sec.pickle'):
            targets.append(l)
    for p, t in feature_tasks:
        for f in ['E4__ACC_X.pickle', 'E4__ACC_Y.pickle', 'E4__ACC_Z.pickle']:
            acc_p = f'EPIStress/{p}/Lab/Labeled/{t}/{f}'
            if acc_p in entries:
                targets.append(acc_p)
    for task in ['relaxation_video', 'arithmetix_hard']:
        for sig in ['Muse__RAW_AF7.pickle', 'E4__BVP.pickle', 'E4__EDA.pickle']:
            p_raw = f'EPIStress/ES140/Lab/Labeled/{task}/{sig}'
            if p_raw in entries:
                targets.append(p_raw)

    targets = sorted(list(set(targets)))
    dest_root = 'stress-detection/data'
    missing = [t for t in targets if not (os.path.exists(os.path.join(dest_root, t)) and os.path.getsize(os.path.join(dest_root, t)) > 0)]
    print(f"Total targets: {len(targets)}, Remaining to download: {len(missing)}", flush=True)

    def fetch_and_save(target_path):
        dest_path = os.path.join(dest_root, target_path)
        if os.path.exists(dest_path) and os.path.getsize(dest_path) > 0:
            return True
        entry = entries[target_path]
        req_start = entry['local_hdr_off']
        req_end = req_start + 256 + entry['comp_size']
        for attempt in range(8):
            try:
                time.sleep(0.48)  # ~125 requests/min keeps strictly within Zenodo limit of 133/min
                req = urllib.request.Request('https://zenodo.org/api/records/16407549/files/EPIStress.zip/content',
                                             headers={'Range': f'bytes={req_start}-{req_end}'})
                with urllib.request.urlopen(req, timeout=25) as resp:
                    raw = resp.read()
                sig, ver, flags, method, time_v, date, crc, c_sz, u_sz, fn_len, ex_len = struct.unpack('<4sHHHHHIIIHH', raw[:30])
                raw_data = raw[30 + fn_len + ex_len : 30 + fn_len + ex_len + entry['comp_size']]
                if entry['method'] == 0:
                    data = raw_data
                elif entry['method'] == 8:
                    data = zlib.decompress(raw_data, -15)
                else:
                    return False
                os.makedirs(os.path.dirname(dest_path), exist_ok=True)
                with open(dest_path, 'wb') as out:
                    out.write(data)
                return True
            except urllib.error.HTTPError as e:
                if e.code == 429:
                    retry_sec = float(e.headers.get('Retry-After', 5))
                    reset_time = e.headers.get('x-ratelimit-reset')
                    if reset_time:
                        wait = max(int(reset_time) - int(time.time()) + 1, retry_sec)
                    else:
                        wait = retry_sec
                    print(f"Rate limited (429), pausing {wait:.1f}s...", flush=True)
                    time.sleep(wait)
                else:
                    time.sleep(1.0)
            except Exception as e:
                time.sleep(1.0)
        print(f"Failed to download {target_path}", flush=True)
        return False

    t0 = time.time()
    completed = 0
    for idx, m in enumerate(missing, 1):
        ok = fetch_and_save(m)
        if ok:
            completed += 1
        if idx % 25 == 0 or idx == len(missing):
            elapsed = time.time() - t0
            rate = idx / max(elapsed, 0.1)
            eta = (len(missing) - idx) / max(rate, 0.01)
            print(f"Progress: {idx}/{len(missing)} ({idx/len(missing)*100:.1f}%) - ETA: {eta:.0f}s", flush=True)

    print(f"Download completed in {time.time()-t0:.1f}s!", flush=True)

if __name__ == '__main__':
    main()
