# 目录
1. 安装官网的docker
2. 安装NVIDIA docker
3. 注册阿里云镜像
4. 下载docker镜像
5. 加入docker用户组
---

# 在Ubuntu服务器上安装Docker 
    sudo apt-get update
    sudo apt-get install apt-transport-https ca-certificates curl software-properties-common
    curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -
    sudo apt-key fingerprint 0EBFCD88
    sudo add-apt-repository "deb [arch=amd64] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable"
    sudo apt-get update
#### 安装docker
`sudo apt-get install docker-ce`
这里在线安装下载太慢，先下载好安装文件，再
`sudo dpkg -i docker-ce_18.03.1_ce-0_ubuntu_amd64.deb`
以上，Docker就安装好了，运行`docker version`或`docker info`验证是否安装成功。

---
#安装NVIDIA docker
    curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
    distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
    curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
    sudo apt-get update
    sudo apt-get install -y nvidia-docker2
    sudo pkill -SIGHUP dockerd
以上，NVIDIA docker就安装好了，可以测试： 
`docker run --runtime=nvidia --rm nvidia/cuda nvidia-smi`
---
#注册阿里云镜像仓库
1. [登陆阿里云链接](https://dev.aliyun.com/search.html)
2. 注册阿里云账号
3. 管理中心→镜像加速器


    sudo mkdir -p /etc/docker
```
sudo tee /etc/docker/daemon.json <<-'EOF'
{
  "registry-mirrors": ["https://pj7n**你的专属地址**p9ua.mirror.aliyuncs.com"]
}
EOF
```
    sudo systemctl daemon-reload
    sudo systemctl restart docker
---
#Docker操作命令
- 下载镜像：
`docker search 镜像关键词`
`docker pull 镜像名称`
- 查看已下载的docker镜像：
`docker images`
- 删除镜像：
`docker rm 镜像`
- 创建nvidia-docker容器并运行：
`nvidia-docker run -itd --name 自定义的容器名称 -v /home/hejia:/mnt(挂载进去的本机目录:挂载的目录) 镜像ID /bin/bash`
- 查看容器：
`docker ps -a`
- 进入容器：
`nvidia-docker exec -it 开启中的容器ID /bin/bash`
- 退出容器：
`exit`或`Ctrl+D`
- 删除容器：
`docker rm 未在运行(Exited)的容器ID`
- 运行容器：
`nvidia-docker start 容器ID`
---------------------------------------
- 下载NGC镜像:
`docker login nvcr.io`
输入用户名和密码：
Username: `$oauthtoken`
Password: `ZmhmYTBxaDlycTFsb2ttaW92NHBwdDM3YWY6ODc1MjI1YWMtMWI4ZS00NjMyLWI3MmQtMTZmNmJlNDZiODlj`
`docker pull nvcr.io/nvidia/tensorflow:18.04-py3（镜像名称） `

- 把用户加入 Docker 用户组:
`sudo usermod -aG docker 用户名`
- 保存镜像为文件:
`docker save -o 要保存的文件名  要保存的镜像`
- 从文件载入镜像:
`docker load --input 文件`
或者
`docker load < 文件名`

- 卸载Decker CE：
```
sudo apt-get purge docker-ce
sudo rm -rf /var/lib/docker
```
