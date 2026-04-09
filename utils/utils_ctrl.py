import os

# Avoid rare OpenMP runtime duplication issues in some environments.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from email.message import EmailMessage
import smtplib
import time
import random


def send_email_with_attachment(subject, body, filename, sender, password, recipient):
    msg = EmailMessage()
    msg['Subject'] = subject
    msg['From'] = sender
    msg['To'] = recipient
    msg.set_content(body)

    # 读取文件并附加
    with open(filename, 'rb') as f:
        file_data = f.read()
        file_name = os.path.basename(filename)
    msg.add_attachment(file_data, maintype='application', subtype='octet-stream', filename=file_name)

    # 连接 SMTP 服务器发送邮件
    with smtplib.SMTP('smtp.163.com', 25) as smtp:
        smtp.ehlo()
        smtp.starttls()
        smtp.ehlo()
        smtp.login(sender, password)  # 注意 password 是“授权码”而不是登录密码
        smtp.send_message(msg)
    print("email sent")
    time.sleep(5 + random.randint(1, 10))  # Pause for 5 seconds
