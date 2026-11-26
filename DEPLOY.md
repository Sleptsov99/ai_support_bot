# Выкладка на ai.avalink.co

Схема: бот в Docker-контейнере слушает только `127.0.0.1:8000`. Снаружи к нему ведёт nginx на сервере:
HTTPS (Let's Encrypt), пароль на демо, лимит 30 сообщений в минуту с одного IP (каждое сообщение стоит токенов).
Несколько тестировщиков работают параллельно: у каждой вкладки свой чат, медленный ответ одному не тормозит других.
Процесс один намеренно: состояние диалогов в памяти (для пилота хватает, для продакшена нужен Redis/Postgres).

Сервер: `root@159.195.248.101`, папка проекта на сервере `/opt/ai_support_bot`.

## 0. DNS

Домен уже заведён и стоит за Cloudflare (`dig +short ai.avalink.co` выдаёт 188.114.x.x: это адреса Cloudflare,
настоящий IP сервера за ними скрыт). Ничего делать не нужно. Реальный IP клиентов nginx берёт из заголовка
Cloudflare (настраивает `setup_server.sh`), поэтому лимит сообщений считается по каждому пользователю.

## 1. Подготовить сервер (один раз)

С ноутбука, из папки проекта:

```bash
scp -i ~/.ssh/id_ed25519 deploy/setup_server.sh root@159.195.248.101:/root/
ssh -i ~/.ssh/id_ed25519 root@159.195.248.101 'bash /root/setup_server.sh'
```

Ставит Docker, nginx, certbot, включает фаервол (открыты только SSH, 80 и 443), создаёт `/opt/ai_support_bot`.
Рассчитано на Ubuntu/Debian.

## 2. Залить код

```bash
SSH_KEY=~/.ssh/id_ed25519 ./deploy/deploy.sh
```

В первый раз он скопирует файлы и остановится с сообщением «No .env on the server»: это нормально, см. шаг 3.

## 3. Ключи на сервере

```bash
ssh -i ~/.ssh/id_ed25519 root@159.195.248.101
cd /opt/ai_support_bot
cp .env.example .env
nano .env            # вписать DEEPSEEK_API_KEY=..., LLM_PROVIDER=deepseek
chmod 600 .env
```

`.env` живёт только на сервере, `deploy.sh` его не трогает и не перезаписывает.

## 4. Запустить бота

```bash
docker compose up -d --build
docker compose ps                       # STATUS: Up (healthy) через ~30 секунд
curl -s 127.0.0.1:8000/health          # {"status":"ok"}
docker compose logs --tail 20           # строка "LLM: deepseek"
```

## 5. nginx и пароль для тестировщиков

```bash
cp deploy/ai-bot-proxy.conf /etc/nginx/snippets/ai-bot-proxy.conf
cp deploy/nginx-ai.avalink.co.conf /etc/nginx/sites-available/ai.avalink.co
ln -sf /etc/nginx/sites-available/ai.avalink.co /etc/nginx/sites-enabled/ai.avalink.co
htpasswd -c /etc/nginx/.htpasswd-ai tester   # придумать пароль; ещё логины: htpasswd /etc/nginx/.htpasswd-ai anna
nginx -t && systemctl reload nginx
```

## 6. HTTPS

Сначала проверить, что Cloudflare действительно ведёт на этот сервер (с ноутбука, логин и пароль из шага 5):

```bash
curl -s -u tester:ПАРОЛЬ http://ai.avalink.co/health     # {"status":"ok"}
```

Затем сертификат на сервере, **без** `--redirect`: на HTTPS клиентов переводит сам Cloudflare,
а редирект на сервере при режиме Flexible даёт бесконечный цикл.

```bash
certbot --nginx -d ai.avalink.co -m ВАШ_EMAIL --agree-tos -n
```

После этого в панели Cloudflare (SSL/TLS → Overview) поставить режим **Full (strict)**: тогда трафик
шифруется и между Cloudflare и сервером.

Если certbot ругается на challenge (в Cloudflare включено «Always Use HTTPS», а на сервере ещё нет 443),
попросить владельца домена на пару минут выключить оранжевое облачко у записи `ai` (DNS only),
выпустить сертификат и включить обратно. Продление дальше работает и через Cloudflare.

Готово: https://ai.avalink.co/demo (браузер спросит логин и пароль из шага 5).

## Обновление

С ноутбука закинуть новый архив и пересобрать на сервере:

```bash
scp -i ~/.ssh/id_ed25519 ~/Downloads/ai_support_bot.zip root@159.195.248.101:/root/
ssh -i ~/.ssh/id_ed25519 root@159.195.248.101
unzip -o /root/ai_support_bot.zip -d /opt/ && cd /opt/ai_support_bot && docker compose up -d --build
```

`.env` и логи при этом не трогаются. Либо одной командой с ноутбука: `SSH_KEY=~/.ssh/id_ed25519 ./deploy/deploy.sh`.

Перезапуск сбрасывает открытые демо-диалоги, тестировщикам нужно нажать «Новый чат».

## Логи и мониторинг

**Страница статистики:** https://ai.avalink.co/admin/stats (тот же пароль, что у демо). Обновляется каждые 30 секунд,
окна «час / сутки / с запуска»: диалоги и ходы, доля и причины передачи на L1, найден ли ответ, время ответа
(среднее и p95), ошибки модели, запасные ответы, непройденный фактчек, токены, доля кэша, расходы в $ (оценка
по ценам из `.env`), темы, языки, последние 50 ходов с текстами и последние ошибки. Если модель перестала отвечать,
вверху появляется красная плашка. Те же данные в JSON: `/admin/stats.json`.

Статистика на странице считается с последнего запуска контейнера. Полная история лежит в файлах:

```bash
docker compose exec bot ls logs                                   # turns-ДАТА.jsonl, errors-ДАТА.jsonl
docker compose exec bot tail -n 20 logs/turns-$(date -u +%F).jsonl  # последние ходы
docker compose exec bot cat logs/errors-$(date -u +%F).jsonl        # ошибки за сегодня
docker compose cp bot:/app/logs ./bot-logs                         # выгрузить всё на сервер
docker compose logs -f | grep ' turn '                             # ходы в реальном времени
```

Одна строка = один ход бота: чат, язык, страна, тема, сигнал, что сделал бот, причина передачи на L1, время,
вызовы модели, токены, тексты клиента и бота (длинные номера карт и телефонов замаскированы до последних 4 цифр).
Файлы старше 30 дней удаляются сами (`LOG_KEEP_DAYS`), тексты можно не писать: `LOG_TEXTS=0` в `.env`.

## Полезное

```bash
docker compose logs -f              # логи в реальном времени
docker compose restart              # перезапуск
docker compose down                 # остановить
```

Подключение к настоящему Consultant: заполнить `CONSULTANT_BOT_TOKEN` и `WEBHOOK_SECRET` в `.env`,
раскомментировать блок `/webhook/` в конфиге nginx. Webhook в Consultant: `https://ai.avalink.co/webhook/<WEBHOOK_SECRET>`.
Сейчас контейнер запускает демо-режим, для боевого режима нужна отдельная команда запуска (`support_bot.app:create_app`).
