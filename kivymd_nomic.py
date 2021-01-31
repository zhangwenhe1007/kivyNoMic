#!/usr/bin/python3

from kivymd.app import MDApp
from kivymd.uix.screen import Screen
from kivymd.uix.button import MDRectangleFlatButton, MDFlatButton, MDFloatingActionButton
from kivymd.uix.dialog import MDDialog
from kivy.lang import Builder
from helpers import num_helper, sms_helper
from call_location import call_location
from sms_testing import message_rating


class Alerte_Anti_Arnaqueurs(MDApp):

    def build(self):

        #Build root widget, that is the core application or interface

        self.theme_cls.primary_palette = 'Green'

        screen = Screen()
        #Create a screen variable and add elements to the Screen to display them

        btn_num = MDRectangleFlatButton(text='Enter', pos_hint={'center_x': 0.5, 'center_y': 0.65},
                                        on_release=self.get_data_num)
        screen.add_widget(btn_num)

        btn_sms = MDRectangleFlatButton(text='Enter', pos_hint={'center_x': 0.5, 'center_y': 0.45},
                                        on_release=self.get_data_sms)
        screen.add_widget(btn_sms)

        self.num = Builder.load_string(num_helper)
        screen.add_widget(self.num)

        self.sms = Builder.load_string(sms_helper)
        screen.add_widget(self.sms)

        return screen

#functions that activate on button release/ on button press

    def get_data_num(self, obj):

        if self.num.text is "":
            check_string = "Please enter a number"
        else:
            check_string = call_location((self.num.text))

        close_btn = MDFlatButton(text='Close', on_release=self.close_dialog)

        self.dialog = MDDialog(title='Verification', text=check_string,
                               size_hint=(0.7, 1),
                               buttons=[close_btn])
        self.dialog.open()

    def get_data_sms(self, obj):


        if self.sms.text is "":
            check_string = "Please enter the message"
        else:
            check_string = message_rating(self.sms.text)

        close_btn = MDFlatButton(text='Close', on_release=self.close_dialog)

        self.dialog = MDDialog(title='Verification', text=check_string,
                               size_hint=(0.7, 1),
                               buttons=[close_btn])
        self.dialog.open()

    def close_dialog(self, obj):
        self.dialog.dismiss()

if __name__ == '__main__':
    Alerte_Anti_Arnaqueurs().run()
