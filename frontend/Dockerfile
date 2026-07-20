# Production build: Vite static bundle diserve nginx.
# VITE_API_BASE_URL di-bake saat build (Vite env = build-time, bukan runtime),
# diambil dari .env yg ikut ter-copy ke build context (lihat .dockerignore).
# node:20-slim (glibc), BUKAN alpine: lock file ini dibuat di Windows dan
# tidak menyimpan varian optional-dep musl milik Rollup ("npm ci" gagal di
# alpine, "Missing: @emnapi/core from lock file").
# `npm install` bukan `npm ci` (fix 18-Jul, ketemu lagi pas nambah jsPDF):
# npm ci MEWAJIBKAN lock file persis sinkron platform target build (linux),
# tapi lock file di-generate di Windows host -> tiap kali nambah dependency
# baru yg py optional native binding (rollup, html2canvas/@napi-rs/canvas
# dsb), npm di Windows "up to date"-in lock file versinya sendiri, BUKAN
# varian linux -> npm ci di Docker gagal lagi. npm install lebih toleran
# (resolve ulang optional deps sesuai platform build saat itu), trade-off
# terima drpd terus2an gagal tiap nambah dependency baru.
FROM node:20-slim AS build
WORKDIR /app
COPY package.json package-lock.json ./
RUN npm install
COPY . .
RUN npm run build

FROM nginx:1.27-alpine
COPY --from=build /app/dist /usr/share/nginx/html
COPY nginx.conf /etc/nginx/conf.d/default.conf
EXPOSE 80
