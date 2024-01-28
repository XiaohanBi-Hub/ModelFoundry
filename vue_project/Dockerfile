# 使用特定版本的Node.js镜像
FROM node:16.20.2

# 设置工作目录
WORKDIR /app

# 复制package.json和package-lock.json
COPY package*.json ./

# 安装项目依赖
RUN npm install

# 复制其他前端代码
COPY . .

RUN npm run build

# 暴露你的前端服务需要的端口，例如8081
EXPOSE 8080
EXPOSE 8081

# 设置启动命令，这取决于你是如何启动你的前端服务的
# CMD [ "bash" ]
# CMD ["npm", "start"]
CMD tail -f /dev/null

