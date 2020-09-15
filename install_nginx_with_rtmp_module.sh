mkdir nginx_install
cd nginx_install

wget http://nginx.org/download/nginx-1.19.2.tar.gz
wget http://openssl.org/source/openssl-1.1.1g.tar.gz
wget https://ftp.pcre.org/pub/pcre/pcre-8.44.tar.gz
wget https://zlib.net/zlib-1.2.11.tar.gz
wget https://github.com/arut/nginx-rtmp-module/archive/master.zip

tar -zxvf nginx-1.19.2.tar.gz
tar -zxvf openssl-1.1.1g.tar.gz
tar -zxvf pcre-8.44.tar.gz
tar -zxvf zlib-1.2.11.tar.gz
unzip master.zip

cd nginx-1.19.2/

./configure --with-zlib=../zlib-1.2.11 --with-pcre=../pcre-8.44 --with-openssl=../openssl-1.1.1g --with-http_ssl_module  --with-debug --add-module=../nginx-rtmp-module-master --prefix=/usr/local/nginx --user=www-data --group=www-data

make
make install

wget https://raw.github.com/JasonGiedymin/nginx-init-ubuntu/master/nginx -O /etc/init.d/nginx
chmod +x /etc/init.d/nginx
update-rc.d -f nginx defaults
service nginx status

cp -f nginx.conf /usr/local/nginx/conf/
mkdir -pZ /data/hdd/kkb/www/live
chown -R www-data:www-data /data/hdd/kkb/www
service nginx start