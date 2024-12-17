# Construye el sitio estatico
FROM python:3.9-slim AS builder
RUN pip install mkdocs-material
WORKDIR /app
COPY . .
RUN mkdocs build
#Usar el servidor nginx
FROM nginx:latest
#Copiar la config de nginx
COPY nginx.conf /etc/nginx/conf.d/default.conf
#Copiar la Wiki a el directorio de nginx
COPY --from=builder /app/site /usr/share/nginx/html

#Exponer en el puerto http
EXPOSE 80