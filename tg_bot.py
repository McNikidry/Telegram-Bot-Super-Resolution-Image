from telegram.ext import Updater
import logging
from telegram import Update
from telegram.ext import CallbackContext
from telegram.ext import CommandHandler, MessageHandler
from telegram.ext.filters import Filters
import numpy as np
from skimage import io
from torchvision import transforms
from inference import inference
import torch
from model import Generator
from torchvision.utils import save_image

updater = Updater(token = '5031776824:AAFbKmZVwAr0Va-7OaH6G0tvssEf8PvmF-s', use_context = True)

dispatcher = updater.dispatcher

logging.basicConfig(format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    level = logging.INFO)

start_text = "Hi! I'm a bot which can help you to increase your image resolution. " \
             "Please, send me a picture, which quality you want to improve."

path_to_model = 'weights/srResNet.pt'
path_to_image = 'images'

def start(update: Update, context: CallbackContext):
    context.bot.send_message(chat_id = update.effective_chat.id, text = start_text)


def process_image(update: Update, context: CallbackContext):
    chat_id = update.message.chat_id
    file = context.bot.get_file(update.message.photo[-1].file_id)
    test_im = io.imread(file['file_path'])
    lr_image_tensor = transforms.functional.to_tensor(test_im).unsqueeze(0)
    output = inference(path_to_model, lr_image_tensor)
    output = output.squeeze()
    save_image(output, 'images/res_image.png')
    context.bot.send_photo(chat_id, open('images/res_image.png', 'rb'))


photo_handler = MessageHandler(Filters.photo, process_image)
start_handler = CommandHandler('start', start)
dispatcher.add_handler(start_handler)
dispatcher.add_handler(photo_handler)
updater.start_polling()
