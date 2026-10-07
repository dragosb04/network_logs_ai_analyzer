import datetime
import random

lines_count = 2000
start_time = datetime.datetime(2026, 10, 7, 10, 15, 30)

templates = [
    # (Level, Template, params)
    ("INFO", "Service started", None),
    ("INFO", "Worker process spawned PID {pid}", "pid"),
    ("INFO", "Health check OK on internal interface", None),
    ("INFO", "Fallback to cached update list", None),
    ("INFO", "Heartbeat received from agent {agent}", "agent"),
    ("INFO", "Cleaned up {num} temporary files", "num"),
    ("WARNING", "Retrying connection to {ip}:443", "ip"),
    ("WARNING", "Disk usage at {disk}%", "disk"),
    ("WARNING", "Memory usage reached {mem}%", "mem"),
    ("WARNING", "Slow query detected: {ms}ms", "ms"),
    ("ERROR", "Connection refused to {ip}:443", "ip"),
    ("ERROR", "DNS lookup failed for {domain}", "domain"),
    ("ERROR", "Login failed for user {user} password={pwd}", "auth"),
    ("ERROR", "Cannot write to /var/data/cache: No space left on device", None),
    ("ERROR", "Timeout waiting for response from {ip}:443", "ip"),
    ("ERROR", "Database connection pool exhausted", None),
]

# Greutăți/probabilități pentru a reflecta un log realist
weights = [1, 5, 20, 4, 30, 5, 25, 6, 6, 8, 30, 15, 10, 8, 12, 5]

ips = ["10.0.0.5", "10.0.0.12", "192.168.1.50", "172.16.0.4"]
domains = ["update.example.com", "api.example.com", "auth.internal.net"]
users = ["admin@example.com", "operator@example.com", "root@example.com", "service_user@example.com"]
passwords = ["hunter2", "admin123", "secretPass!", "toor"]

current_time = start_time
output_lines = []

for i in range(lines_count):
    if i == 0:
        msg = "INFO Service started"
    elif i == lines_count - 1:
        msg = "INFO Service stopped"
    else:
        lvl, tpl, ptype = random.choices(templates, weights=weights)[0]
        if ptype is None:
            text = tpl
        elif ptype == "ip":
            text = tpl.format(ip=random.choice(ips))
        elif ptype == "domain":
            text = tpl.format(domain=random.choice(domains))
        elif ptype == "auth":
            text = tpl.format(user=random.choice(users), pwd=random.choice(passwords))
        elif ptype == "disk":
            text = tpl.format(disk=random.randint(90, 99))
        elif ptype == "mem":
            text = tpl.format(mem=random.randint(85, 96))
        elif ptype == "pid":
            text = tpl.format(pid=random.randint(1000, 9999))
        elif ptype == "agent":
            text = tpl.format(agent=random.randint(1, 10))
        elif ptype == "num":
            text = tpl.format(num=random.randint(5, 120))
        elif ptype == "ms":
            text = tpl.format(ms=random.randint(1200, 8500))
        
        msg = f"{lvl} {text}"
    
    timestamp = current_time.strftime("%Y-%m-%d %H:%M:%S")
    output_lines.append(f"{timestamp} {msg}")
    
    # Avansăm timpul cu 1 până la 4 secunde între mesaje
    current_time += datetime.timedelta(seconds=random.randint(1, 4))

with open("datasets.log", "w", encoding="utf-8") as f:
    f.write("\n".join(output_lines) + "\n")

print(f"Fișierul generat conține exact {len(output_lines)} linii.")