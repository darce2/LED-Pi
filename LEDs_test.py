#!/usr/bin/env python
# -*- coding: utf-8 -*-
#

#This imports the Drivers for LED strips of type LPD8806 and sets it up
from bibliopixel.drivers.LPD8806 import *
driver = DriverLPD8806(num = 32, c_order= ChannelOrder.GRB, SPISpeed = 16) #SPI$

#Import led info and set up
from bibliopixel.led import *
led = LEDStrip(driver)

#Import and setup test animation
from bibliopixel.animation import *
anim1 = StripChannelTest(led)

#Import and setup test animation
#from strip_animation import *
#anim = RainbowCycle(led)

#Turn on LEDs 0-31
try:

        led.fillRGB(r = 255, g = 255, b = 255,start = 0, end = 0)
        led.fillRGB(r = 255, g = 255, b = 255, start = 1, end = 1)
        led.fillRGB(r = 255, g = 255, b = 255, start = 2, end = 2)
        led.fillRGB(r = 255, g = 255, b = 255, start = 3, end = 3)
        led.fillRGB(r = 255, g = 255, b = 255, start = 4, end = 4)
        led.fillRGB(r = 255, g = 255, b = 255, start = 5, end = 5)
        led.fillRGB(r = 255, g = 255, b = 255, start = 6, end = 6)
        led.fillRGB(r = 255, g = 255, b = 255, start = 7, end = 7)
	led.fillRGB(r = 255, g = 255, b = 255, start = 8, end = 8)
        led.fillRGB(r = 255, g = 255, b = 255, start = 9, end = 9)
        led.fillRGB(r = 255, g = 255, b = 255, start = 10, end= 10)
        led.fillRGB(r = 255, g = 255, b = 255, start = 11, end= 11)
        led.fillRGB(r = 255, g = 255, b = 255, start = 12, end= 12)
        led.fillRGB(r = 255, g = 255, b = 255, start = 13, end= 13)
        led.fillRGB(r = 255, g = 255, b = 255, start = 14, end =14)
        led.fillRGB(r = 255, g = 255, b = 255, start = 15, end= 15)
        led.fillRGB(r = 255, g = 255, b = 255, start = 16, end = 16)
        led.fillRGB(r = 255, g = 255, b = 255, start = 17, end = 17)
        led.fillRGB(r = 255, g = 255, b = 255, start = 18, end = 18)
        led.fillRGB(r = 255, g = 255, b = 255, start = 19, end = 19)
        led.fillRGB(r = 255, g = 255, b = 255, start = 20, end = 20)
        led.fillRGB(r = 255, g = 255, b = 255, start = 21, end = 21)
        led.fillRGB(r = 255, g = 255, b = 255, start = 22, end = 22)
        led.fillRGB(r = 255, g = 255, b = 255, start = 23, end = 23)
        led.fillRGB(r = 255, g = 255, b = 255, start = 24, end = 24)
        led.fillRGB(r = 255, g = 255, b = 255, start = 25, end = 25)
        led.fillRGB(r = 255, g = 255, b = 255, start = 26, end = 26)
        led.fillRGB(r = 255, g = 255, b = 255, start = 27, end = 27)
        led.fillRGB(r = 255, g = 255, b = 255, start = 28, end = 28)
        led.fillRGB(r = 255, g = 255, b = 255, start = 29, end = 29)
        led.fillRGB(r = 255, g = 255, b = 255, start = 30, end = 30)
        led.fillRGB(r = 255, g = 255, b = 255, start = 31, end = 31)
        led.fillRGB(r = 255, g = 255, b = 255, start = 32, end = 32)
        led.update()
except KeyboardInterrupt:
        led.all_off()
        led.update()
        
