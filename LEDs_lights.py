#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  This Program aims to combine Lighshow Pi and BilblioPixel to create a  led strip
#That responds to music imput through the usb port


#imports from the New lighshowpi sync lights 

import argparse
import csv
import fcntl
import gzip
import json
import logging
import os
import random
import subprocess
import sys
import wave

import alsaaudio as aa
import fft
import configuration_manager as cm
import decoder
import hardware_controller as hc
import numpy as np

from preshow import Preshow

##imports from scott driscolls hack of the ledstrip_lighshow Pi (duplicates removed)


from struct import unpack
from time import sleep
import time


#imports to make the lights work
from bibliopixel.drivers.LPD8806 import *
from bibliopixel.led import *
from bibliopixel.colors import *
#from bibliopixel.animation import *

#how many lights (32 per strip)
num_lights = 32

#set instances
led_driver_LP = DriverLPD8806(num = num_lights, c_order= ChannelOrder.GRB, SPISpeed = 16) #SPI speed still isn't fully understood but this value seems to work
led = LEDStrip(led_driver_LP)
led.all_off()
#set led brightness (0-255)

#led.masterBrightness(100)
#led.update()

#This writes out light info to the LED strip 
#The colors are simply fading through all the colors######################################
c = 0.0
def display_ledStrip()
	global c
	global columns
	color = wheel_color(int(c))
	c = c + .1
	if c > 384
		c = 0.0
	 led.fil(color, end = num_lights)

###########################################################################

# Configurations - TODO(todd): Move more of this into configuration manager
_CONFIG = cm.CONFIG
_MODE = cm.lightshow()['mode']
_MIN_FREQUENCY = _CONFIG.getfloat('audio_processing', 'min_frequency')
_MAX_FREQUENCY = _CONFIG.getfloat('audio_processing', 'max_frequency')
_RANDOMIZE_PLAYLIST = _CONFIG.getboolean('lightshow', 'randomize_playlist')
try:
    _CUSTOM_CHANNEL_MAPPING = [int(channel) for channel in
                               _CONFIG.get('audio_processing', 'custom_channel_mapping').split(',')]
except:
    _CUSTOM_CHANNEL_MAPPING = 0 ###############THIS COULD BE USEFUL FOR CUSTOME LIGHTS!!!!!!" "
try:
    _CUSTOM_CHANNEL_FREQUENCIES = [int(channel) for channel in
                                   _CONFIG.get('audio_processing',
                                               'custom_channel_frequencies').split(',')]
except:
    _CUSTOM_CHANNEL_FREQUENCIES = 0
try:
    _PLAYLIST_PATH = cm.lightshow()['playlist_path'].replace('$SYNCHRONIZED_LIGHTS_HOME', cm.HOME_DIR)
except: 
    _PLAYLIST_PATH = "/home/pi/music/.playlist"
try:
    _usefm=_CONFIG.get('audio_processing','fm');
    frequency =_CONFIG.get('audio_processing','frequency');
    play_stereo = True
    music_pipe_r,music_pipe_w = os.pipe()	
except:
    _usefm='false'
CHUNK_SIZE = 2048  # Use a multiple of 8 (move this to config)




#########################################################################
def calculate_channel_frequency(min_frequency, max_frequency, custom_channel_mapping,
                                custom_channel_frequencies):
    '''Calculate frequency values for each channel, taking into account custom settings.'''

    # How many channels do we need to calculate the frequency for
    if custom_channel_mapping != 0 and len(custom_channel_mapping) == hc.GPIOLEN:
        logging.debug("Custom Channel Mapping is being used: %s", str(custom_channel_mapping))
        channel_length = max(custom_channel_mapping)
    else:
        logging.debug("Normal Channel Mapping is being used.")
        channel_length = hc.GPIOLEN

    logging.debug("Calculating frequencies for %d channels.", channel_length)
    octaves = (np.log(max_frequency / min_frequency)) / np.log(2)
    logging.debug("octaves in selected frequency range ... %s", octaves)
    octaves_per_channel = octaves / channel_length
    frequency_limits = []
    frequency_store = []

    frequency_limits.append(min_frequency)
    if custom_channel_frequencies != 0 and (len(custom_channel_frequencies) >= channel_length + 1):
        logging.debug("Custom channel frequencies are being used")
        frequency_limits = custom_channel_frequencies
    else:
        logging.debug("Custom channel frequencies are not being used")
        for i in range(1, hc.GPIOLEN + 1):
            frequency_limits.append(frequency_limits[-1]
                                    * 10 ** (3 / (10 * (1 / octaves_per_channel))))
    for i in range(0, channel_length):
        frequency_store.append((frequency_limits[i], frequency_limits[i + 1]))
        logging.debug("channel %d is %6.2f to %6.2f ", i, frequency_limits[i],
                      frequency_limits[i + 1])

    # we have the frequencies now lets map them if custom mapping is defined
    if custom_channel_mapping != 0 and len(custom_channel_mapping) == hc.GPIOLEN:
        frequency_map = []
        for i in range(0, hc.GPIOLEN):
            mapped_channel = custom_channel_mapping[i] - 1
            mapped_frequency_set = frequency_store[mapped_channel]
            mapped_frequency_set_low = mapped_frequency_set[0]
            mapped_frequency_set_high = mapped_frequency_set[1]
            logging.debug("mapped channel: " + str(mapped_channel) + " will hold LOW: "
                          + str(mapped_frequency_set_low) + " HIGH: "
                          + str(mapped_frequency_set_high))
            frequency_map.append(mapped_frequency_set)
        return frequency_map
    else:
        return frequency_store



########################################################################
#Believed to return the power array such that
#So if your sample rate, Fs is say 44.1 kHz and your FFT size, N is 1024, then the FFT output bins are at:

 # 0:   0 * 44100 / 1024 =     0.0 Hz
 # 1:   1 * 44100 / 1024 =    43.1 Hz
 # 2:   2 * 44100 / 1024 =    86.1 Hz
 # 3:   3 * 44100 / 1024 =   129.2 Hz
 ## 4: ...
  #5: ...
     ...
#511: 511 * 44100 / 1024 = 22006.9 Hz

def piff(val, sample_rate):
    '''Return the power array index corresponding to a particular frequency.'''
    return int(CHUNK_SIZE * val / sample_rate)



############################################################################
#now channel frequencies are calculated so now we just need to select the ones we want
#and make sure they make sense for the color we need....



########Calculate_levels function skipped in new version
def calculate_levels(data, sample_rate, frequency_limits):
    '''Calculate frequency response for each channel
    
    Initial FFT code inspired from the code posted here:
    http://www.raspberrypi.org/phpBB3/viewtopic.php?t=35838&p=454041
    
    Optimizations from work by Scott Driscoll:
    http://www.instructables.com/id/Raspberry-Pi-Spectrum-Analyzer-with-RGB-LED-Strip-/
    '''

    # create a numpy array. This won't work with a mono file, stereo only.
    data_stereo = np.frombuffer(data, dtype=np.int16)
    data = np.empty(len(data) / 4)  # data has two channels and 2 bytes per channel
    data[:] = data_stereo[::2]  # pull out the even values, just using left channel

    # if you take an FFT of a chunk of audio, the edges will look like
    # super high frequency cutoffs. Applying a window tapers the edges
    # of each end of the chunk down to zero.
    window = np.hanning(len(data))
    data = data * window

    # Apply FFT - real data
    fourier = np.fft.rfft(data)

    # Remove last element in array to make it the same size as CHUNK_SIZE
    fourier = np.delete(fourier, len(fourier) - 1)

    # Calculate the power spectrum
    power = np.abs(fourier) ** 2

    matrix = [0 for i in range(hc.GPIOLEN)]
    for i in range(hc.GPIOLEN):
        # take the log10 of the resulting sum to approximate how human ears perceive sound levels
        matrix[i] = np.log10(np.sum(power[piff(frequency_limits[i][0], sample_rate)
                                          :piff(frequency_limits[i][1], sample_rate):1]))

    return matrix

#This gets the audio in from the aux cord to usb inverter 
##############################################################################
#In old version there is a main where it is song to play 
#in new version simply a function named 
######

## Audio in then play song
def audio_in():
    '''Control the lightshow from audio coming in from a USB audio card'''
    sample_rate = cm.lightshow()['audio_in_sample_rate']
    input_channels = cm.lightshow()['audio_in_channels']

    # Open the input stream from default input device
    stream = aa.PCM(aa.PCM_CAPTURE, aa.PCM_NORMAL, cm.lightshow()['audio_in_card'])
    stream.setchannels(input_channels)
    stream.setformat(aa.PCM_FORMAT_S16_LE) # Expose in config if needed
    stream.setrate(sample_rate)
    stream.setperiodsize(CHUNK_SIZE)
         
    logging.debug("Running in audio-in mode - will run until Ctrl+C is pressed")
    print "Running in audio-in mode, use Ctrl+C to stop"
    try:
        hc.initialize()
        frequency_limits = calculate_channel_frequency(_MIN_FREQUENCY,
                                                       _MAX_FREQUENCY,
                                                       _CUSTOM_CHANNEL_MAPPING,
                                                       _CUSTOM_CHANNEL_FREQUENCIES)

        # Start with these as our initial guesses - will calculate a rolling mean / std 
        # as we get input data.
        mean = [12.0 for _ in range(hc.GPIOLEN)]
        std = [0.5 for _ in range(hc.GPIOLEN)]
        recent_samples = np.empty((250, hc.GPIOLEN))
        num_samples = 0
    
        # Listen on the audio input device until CTRL-C is pressed
        while True:            
            l, data = stream.read()
            
            if l:
                try:
                    matrix = fft.calculate_levels(data, CHUNK_SIZE, sample_rate, frequency_limits, input_channels)
                    if not np.isfinite(np.sum(matrix)):
                        # Bad data --- skip it
                        continue
                except ValueError as e:
                    # TODO(todd): This is most likely occuring due to extra time in calculating
                    # mean/std every 250 samples which causes more to be read than expected the
                    # next time around.  Would be good to update mean/std in separate thread to
                    # avoid this --- but for now, skip it when we run into this error is good 
                    # enough ;)
                    logging.debug("skipping update: " + str(e))
                    continue

                update_lights(matrix, mean, std)

                # Keep track of the last N samples to compute a running std / mean
                #
                # TODO(todd): Look into using this algorithm to compute this on a per sample basis:
                # http://www.johndcook.com/blog/standard_deviation/                
                if num_samples >= 250:
                    no_connection_ct = 0
                    for i in range(0, hc.GPIOLEN):
                        mean[i] = np.mean([item for item in recent_samples[:, i] if item > 0])
                        std[i] = np.std([item for item in recent_samples[:, i] if item > 0])
                        
                        # Count how many channels are below 10, if more than 1/2, assume noise (no connection)
                        if mean[i] < 10.0:
                            no_connection_ct += 1
                            
                    # If more than 1/2 of the channels appear to be not connected, turn all off
                    if no_connection_ct > hc.GPIOLEN / 2:
                        logging.debug("no input detected, turning all lights off")
                        mean = [20 for _ in range(hc.GPIOLEN)]
                    else:
                        logging.debug("std: " + str(std) + ", mean: " + str(mean))
                    num_samples = 0
                else:
                    for i in range(0, hc.GPIOLEN):
                        recent_samples[num_samples][i] = matrix[i]
                    num_samples += 1
 
    except KeyboardInterrupt:
        pass
    finally:
        print "\nStopping"
        hc.clean_up()

# TODO(todd): Refactor more of this to make it more readable / modular.
def play_song():
    '''Play the next song from the play list (or --file argument).'''
    song_to_play = int(cm.get_state('song_to_play', 0))
    play_now = int(cm.get_state('play_now', 0))

    # Arguments
    parser = argparse.ArgumentParser()
    filegroup = parser.add_mutually_exclusive_group()
    filegroup.add_argument('--playlist', default=_PLAYLIST_PATH,
                           help='Playlist to choose song from.')
    filegroup.add_argument('--file', help='path to the song to play (required if no'
                           'playlist is designated)')
    parser.add_argument('--readcache', type=int, default=1,
                        help='read light timing from cache if available. Default: true')
    args = parser.parse_args()

    # Make sure one of --playlist or --file was specified
    if args.file == None and args.playlist == None:
        print "One of --playlist or --file must be specified"
        sys.exit()

    # Initialize Lights
    hc.initialize()

    # Handle the pre-show
    if not play_now:
        result = Preshow().execute()
        if result == Preshow.PlayNowInterrupt:
            play_now = True

    # Determine the next file to play
    song_filename = args.file
    if args.playlist != None and args.file == None:
        most_votes = [None, None, []]
        current_song = None
        with open(args.playlist, 'rb') as playlist_fp:
            fcntl.lockf(playlist_fp, fcntl.LOCK_SH)
            playlist = csv.reader(playlist_fp, delimiter='\t')
            songs = []
            for song in playlist:
                if len(song) < 2 or len(song) > 4:
                    logging.error('Invalid playlist.  Each line should be in the form: '
                                 '<song name><tab><path to song>')
                    sys.exit()
                elif len(song) == 2:
                    song.append(set())
                else:
                    song[2] = set(song[2].split(','))
                    if len(song) == 3 and len(song[2]) >= len(most_votes[2]):
                        most_votes = song
                songs.append(song)
            fcntl.lockf(playlist_fp, fcntl.LOCK_UN)

        if most_votes[0] != None:
            logging.info("Most Votes: " + str(most_votes))
            current_song = most_votes

            # Update playlist with latest votes
            with open(args.playlist, 'wb') as playlist_fp:
                fcntl.lockf(playlist_fp, fcntl.LOCK_EX)
                writer = csv.writer(playlist_fp, delimiter='\t')
                for song in songs:
                    if current_song == song and len(song) == 3:
                        song.append("playing!")
                    if len(song[2]) > 0:
                        song[2] = ",".join(song[2])
                    else:
                        del song[2]
                writer.writerows(songs)
                fcntl.lockf(playlist_fp, fcntl.LOCK_UN)

        else:
            # Get a "play now" requested song
            if play_now > 0 and play_now <= len(songs):
                current_song = songs[play_now - 1]
            # Get random song
            elif _RANDOMIZE_PLAYLIST:
                current_song = songs[random.randint(0, len(songs) - 1)]
            # Play next song in the lineup
            else:
                song_to_play = song_to_play if (song_to_play <= len(songs) - 1) else 0
                current_song = songs[song_to_play]
                next_song = (song_to_play + 1) if ((song_to_play + 1) <= len(songs) - 1) else 0
                cm.update_state('song_to_play', next_song)

        # Get filename to play and store the current song playing in state cfg
        song_filename = current_song[1]
        cm.update_state('current_song', songs.index(current_song))

    song_filename = song_filename.replace("$SYNCHRONIZED_LIGHTS_HOME", cm.HOME_DIR)

    # Ensure play_now is reset before beginning playback
    if play_now:
        cm.update_state('play_now', 0)
        play_now = 0

    # Initialize FFT stats
    matrix = [0 for _ in range(hc.GPIOLEN)]
    offct = [0 for _ in range(hc.GPIOLEN)] #### from old

    # Set up audio
    if song_filename.endswith('.wav'):
        musicfile = wave.open(song_filename, 'r')
    else:
        musicfile = decoder.open(song_filename)

    sample_rate = musicfile.getframerate()
    num_channels = musicfile.getnchannels()

    if _usefm=='true':
        logging.info("Sending output as fm transmission")
        with open(os.devnull, "w") as dev_null:
            fm_process = subprocess.Popen(["sudo",cm.HOME_DIR + "/bin/pifm","-",str(frequency),"44100", "stereo" if play_stereo else "mono"], stdin=music_pipe_r, stdout=dev_null)
    else:
        output = aa.PCM(aa.PCM_PLAYBACK, aa.PCM_NORMAL)
        output.setchannels(num_channels)
        output.setrate(sample_rate)
        output.setformat(aa.PCM_FORMAT_S16_LE)
        output.setperiodsize(CHUNK_SIZE)
    
    logging.info("Playing: " + song_filename + " (" + str(musicfile.getnframes() / sample_rate)
                 + " sec)")
    # Output a bit about what we're about to play to the logs
    song_filename = os.path.abspath(song_filename)
    

    cache = []
    cache_found = False
    cache_filename = os.path.dirname(song_filename) + "/." + os.path.basename(song_filename) \
        + ".sync.gz"
    # The values 12 and 1.5 are good estimates for first time playing back (i.e. before we have
    # the actual mean and standard deviations calculated for each channel).
    mean = [12.0 for _ in range(hc.GPIOLEN)]
    std = [1.5 for _ in range(hc.GPIOLEN)]
    if args.readcache:
        # Read in cached fft
        try:
            with gzip.open(cache_filename, 'rb') as playlist_fp:
                cachefile = csv.reader(playlist_fp, delimiter=',')
                for row in cachefile:
                    cache.append([0.0 if np.isinf(float(item)) else float(item) for item in row])
                cache_found = True
                # TODO(todd): Optimize this and / or cache it to avoid delay here
                cache_matrix = np.array(cache)
                for i in range(0, hc.GPIOLEN):
                    std[i] = np.std([item for item in cache_matrix[:, i] if item > 0])
                    mean[i] = np.mean([item for item in cache_matrix[:, i] if item > 0])
                logging.debug("std: " + str(std) + ", mean: " + str(mean))
        except IOError:
            logging.warn("Cached sync data song_filename not found: '" + cache_filename
                         + ".  One will be generated.")

    # Process audio song_filename
    row = 0
    data = musicfile.readframes(CHUNK_SIZE)
    frequency_limits = calculate_channel_frequency(_MIN_FREQUENCY,
                                                   _MAX_FREQUENCY,
                                                   _CUSTOM_CHANNEL_MAPPING,
                                                   _CUSTOM_CHANNEL_FREQUENCIES)

    while data != '' and not play_now:
        if _usefm=='true':
            os.write(music_pipe_w, data)
        else:
            output.write(data)

        # Control lights with cached timing values if they exist
        matrix = None
        if cache_found and args.readcache:
            if row < len(cache):
                matrix = cache[row]
            else:
                logging.warning("Ran out of cached FFT values, will update the cache.")
                cache_found = False

        if matrix == None:
            # No cache - Compute FFT in this chunk, and cache results
            matrix = fft.calculate_levels(data, CHUNK_SIZE, sample_rate, frequency_limits)
            cache.append(matrix)
         

        #blank out the display ### FROM OLD
        led.fill(Color(0,0,0),0,151)
        for i in range(0, hc.GPIOLEN):
            if hc.is_pin_pwm(i):
                # Output pwm, where off is at 0.5 std below the mean
                # and full on is at 0.75 std above the mean.
                
                display_column(i,matrix[i])
                
                #brightness = matrix[i] - mean[i] + 0.5 * std[i]
                #brightness = brightness / (1.25 * std[i])
                #if brightness > 1.0:
                    #brightness = 1.0
                #if brightness < 0:
                    #brightness = 0
                #hc.turn_on_light(i, True, int(brightness * 60))
            else:
                if limit[i] < matrix[i] * _LIMIT_THRESHOLD:
                    limit[i] = limit[i] * _LIMIT_THRESHOLD_INCREASE
                    logging.debug("++++ channel: {0}; limit: {1:.3f}".format(i, limit[i]))
                # Amplitude has reached threshold
                if matrix[i] > limit[i]:
                    hc.turn_on_light(i, True)
                    offct[i] = 0
                else:  # Amplitude did not reach threshold
                    offct[i] = offct[i] + 1
                    if offct[i] > _MAX_OFF_CYCLES:
                        offct[i] = 0
                        limit[i] = limit[i] * _LIMIT_THRESHOLD_DECREASE  # old value 0.8
                    logging.debug("---- channel: {0}; limit: {1:.3f}".format(i, limit[i]))
                    hc.turn_off_light(i, True)


        #send out data to RGB strip
        led.update()
        #read next chunk of data 
        data = musicfil.readframes(CHUNK_SIZE)
        

              #####################################################END OLD

        update_lights(matrix, mean, std)

        # Read next chunk of data from music song_filename
        data = musicfile.readframes(CHUNK_SIZE)
        row = row + 1

        # Load new application state in case we've been interrupted
        cm.load_state()
        play_now = int(cm.get_state('play_now', 0))

    if not cache_found:
        with gzip.open(cache_filename, 'wb') as playlist_fp:
            writer = csv.writer(playlist_fp, delimiter=',')
            writer.writerows(cache)
            logging.info("Cached sync data written to '." + cache_filename
                         + "' [" + str(len(cache)) + " rows]")

    # Cleanup the pifm process
    if _usefm=='true':
        fm_process.kill()

    # We're done, turn it all off and clean up things ;)
    hc.clean_up()

if __name__ == "__main__":
    # Log everything to our log file
    # TODO(todd): Add logging configuration options.
    logging.basicConfig(filename=cm.LOG_DIR + '/music_and_lights.play.dbg',
                        format='[%(asctime)s] %(levelname)s {%(pathname)s:%(lineno)d}'
                        ' - %(message)s',
                        level=logging.DEBUG)

    if cm.lightshow()['mode'] == 'audio-in':
        audio_in()
    else:
        play_song()

