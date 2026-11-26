import struct,time,urllib.request as U,zipfile,zlib
from range_archive import URL

def payload(info):
    start=info.header_offset
    for trial in range(6):
        try:
            req=U.Request(URL+'?segment='+str(start)+'&trial='+str(trial),headers={'Range':f'bytes={start}-{start+info.compress_size+510}'})
            with U.urlopen(req,timeout=25) as res:buf=res.read()
            header=struct.unpack('<IHHHHHIIIHH',buf[:30]);assert header[0]==0x04034b50
            offset=30+header[-1]+header[-2]
            assert buf[30:30+header[-2]].decode()==info.filename
            raw=buf[offset:offset+info.compress_size]
            data=zlib.decompress(raw,-15) if info.compress_type==zipfile.ZIP_DEFLATED else raw
            assert zlib.crc32(data)==info.CRC
            return data
        except Exception:
            if trial==5:raise
            time.sleep(1+trial)
