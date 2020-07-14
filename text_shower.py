import datetime


class TextShower:
    text = ''
    expiration_date = datetime.datetime.now()

    def set_new_text(self, text, expire_in):
        self.text = text
        self.expiration_date = datetime.datetime.now() + datetime.timedelta(milliseconds=expire_in)
        return

    def is_to_show(self):
        return self.expiration_date > datetime.datetime.now()
