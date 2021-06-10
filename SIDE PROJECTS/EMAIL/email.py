import smtplib
import mimetypes
from email.message import EmailMessage

#ctype, encoding = mimetypes.guess_type(filename1)
#ctype, encoding = mimetypes.guess_type(filename2)
#ctype, encoding = mimetypes.guess_type(filename3)
#ctype, encoding = mimetypes.guess_type(filename4)
#ctype, encoding = mimetypes.guess_type(filename5)
#ctype, encoding = mimetypes.guess_type(filename6)
#ctype, encoding = mimetypes.guess_type(filename7)

'''if ctype is None or encoding is not None:
    # No guess could be made, or the file is encoded (compressed), so
    # use a generic bag-of-bits type.
    ctype = 'application/octet-stream'
maintype, subtype = ctype.split('/', 1)
with open(filename1, 'rb') as fp:
    msg.add_attachment(fp.read(),
                       maintype=maintype,
                       subtype=subtype,
                       filename=filename1)
with open(filename2, 'rb') as fp:
    msg.add_attachment(fp.read(),
                       maintype=maintype,
                       subtype=subtype,
                       filename=filename2)
with open(filename3, 'rb') as fp:
    msg.add_attachment(fp.read(),
                       maintype=maintype,
                       subtype=subtype,
                       filename=filename3)
with open(filename4, 'rb') as fp:
    msg.add_attachment(fp.read(),
                       maintype=maintype,
                       subtype=subtype,
                       filename=filename4)
with open(filename4, 'rb') as fp:
    msg.add_attachment(fp.read(),
                       maintype=maintype,
                       subtype=subtype,
                       filename=filename5)
with open(filename4, 'rb') as fp:
    msg.add_attachment(fp.read(),
                       maintype=maintype,
                       subtype=subtype,
                       filename=filename6)
with open(filename4, 'rb') as fp:
    msg.add_attachment(fp.read(),
                       maintype=maintype,
                       subtype=subtype,
                       filename=filename7)'''

msg = EmailMessage()
msg.set_content("Coucou moi-même. Voici le lien que je m'envoie tout seul pour les documents concernant le BP. J'espère que je trouverai mon bonheur... \n\n https://we.tl/t-YVVnDTX3kO \n\n https://we.tl/t-pD2hOmADZQ")


msg['Subject'] = 'Business plan'
msg['From'] = 'Kate.charles.group@gmail.com'
msg['To'] = 'Kate.charles.group@gmail.com' #'GM@css-europe.co.uk'
msg['Cc'] = ''

# Send the message via our own SMTP server.
s = smtplib.SMTP('smtp.free.fr')
s.send_message(msg)
s.quit()    