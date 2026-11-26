import io,urllib.request as u,zipfile
URL='https://dl.dropboxusercontent.com/s/5h3dsjorbrpmjqh/rect1_json.zip'
class Remote(io.RawIOBase):
    def __init__(self):
        super().__init__();self.p=0;self.buf=b'';self.begin=-1
        with u.urlopen(u.Request(URL,headers={'Range':'bytes=0-0'}),timeout=30) as r:self.n=int(r.headers['Content-Range'].split('/')[-1])
    def seekable(self):return True
    def tell(self):return self.p
    def seek(self,k,w=0):self.p=[k,self.p+k,self.n+k][w];return self.p
    def read(self,n=-1):
        if n<0:n=self.n-self.p
        if not n:return b''
        if n>524288:
            from concurrent.futures import ThreadPoolExecutor
            base=self.p; size=128*1024
            def fetch(offset):
                end=min(base+n,base+offset+size)-1
                for attempt in range(4):
                    try:
                        req=u.Request(URL+f'?range={base+offset}&try={attempt}',headers={'Range':f'bytes={base+offset}-{end}'})
                        with u.urlopen(req,timeout=35) as r:chunk=r.read()
                        assert len(chunk)==end-base-offset+1
                        return chunk
                    except Exception:
                        if attempt==3:raise
            with ThreadPoolExecutor(12) as pool:out=b''.join(pool.map(fetch,range(0,n,size)))
            self.p+=n;return out
        if not self.begin<=self.p<=self.p+n<=self.begin+len(self.buf):
            hi=min(self.n,self.p+max(n,131072))
            req=u.Request(URL+'?segment='+str(self.p),headers={'Range':f'bytes={self.p}-{hi-1}'})
            with u.urlopen(req,timeout=40) as r:
                if r.status!=206:raise IOError('Expected HTTP range')
                self.begin=self.p;self.buf=r.read()
        a=self.p-self.begin;out=self.buf[a:a+n];self.p+=len(out);return out
