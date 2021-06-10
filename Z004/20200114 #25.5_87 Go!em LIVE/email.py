import smtplib
import mimetypes
from email.message import EmailMessage

msg['Subject'] = 'Business plan'
msg['From'] = 'jyss@dag26.com'
msg['To'] = 'jyss73@free.fr' #'GM@css-europe.co.uk'
msg['Cc'] = ''

# Send the message via our own SMTP server.
s = smtplib.SMTP('smtp.free.fr')
s.send_message(msg)
s.quit()    