import logging
from typing import Tuple, Dict, Any
from skimage import io
from model import Generator
from denoiser import RidNet
from inference import inference
from torchvision import transforms
from torchvision.utils import save_image
from telegram.ext.filters import Filters
from telegram import InlineKeyboardMarkup, InlineKeyboardButton, Update
from telegram.ext import (
    Updater,
    CommandHandler,
    MessageHandler,
    Filters,
    ConversationHandler,
    CallbackQueryHandler,
    CallbackContext,
)

path_to_gen_model = 'weights/srResNet.pt'
path_to_denoiser_model = 'weights/denoiser_with_gauss_noise.pt'
path_to_image = 'image'

SELECTING_ACTION, IMG_RES, NOISE, IMAGE_RES_NOISE = map(chr, range(4))
IMG_FILE_SR, IMG_FILE_DENOISE, IMG_FILE_SR_DENOISE = map(chr, range(4, 7))
START_OVER = range(7, 8)
RESULT, NEW_MENU = range(8, 10)
END = ConversationHandler.END


def start(update: Update, context: CallbackContext) -> None:
    """Sends a message with three inline buttons attached."""
    start_text = "You may choose on of the options below."

    buttons = [
        [
            InlineKeyboardButton("Increase image resolution", callback_data = str(IMG_RES)),
            InlineKeyboardButton("Reduce noise in image", callback_data = str(NOISE)),
        ],
        [InlineKeyboardButton("Increase image resolution and reduce noise", callback_data = str(IMAGE_RES_NOISE))],
    ]

    keyboard = InlineKeyboardMarkup(buttons)

    # If we're starting over we don't need to send a new message
    if context.user_data.get(START_OVER):

        update.callback_query.answer()
        update.callback_query.edit_message_text(text = start_text, reply_markup = keyboard)

    else:
        update.message.reply_text(
            "Hi! I'm a bot which can help you to increase your image resolution or " \
            "to reduce noise in your image. "
        )
        update.message.reply_text(text = start_text, reply_markup = keyboard)

    context.user_data[START_OVER] = False

    return SELECTING_ACTION


def super_resolution(update: Update, context: CallbackContext) -> str:
    """ Answer to super resolution choice"""
    text = 'Great choice! ' \
           'Please, send me an image, which resolution you want to improve.'

    update.callback_query.answer()
    update.callback_query.edit_message_text(text = text)
    return IMG_FILE_SR


def denoise(update: Update, context: CallbackContext) -> str:
    """Answer to denoise choice"""
    text = 'Great choice! ' \
           'Please, send me an image, which you want to denoise.'

    update.callback_query.answer()
    update.callback_query.edit_message_text(text = text)
    return IMG_FILE_DENOISE


def sr_denoise(update: Update, context: CallbackContext) -> str:
    """Answer to super resolution and denoise choice"""
    text = 'Great choice! ' \
           'Please, send me an image, which resolution you want to improve and then denoise.'

    update.callback_query.answer()
    update.callback_query.edit_message_text(text = text)
    return IMG_FILE_SR_DENOISE


def get_SR_image(update: Update, context: CallbackContext):
    """processes image to SR"""
    chat_id = update.message.chat_id
    file = context.bot.get_file(update.message.photo[-1].file_id)
    test_im = io.imread(file['file_path'])
    lr_image_tensor = transforms.functional.to_tensor(test_im).unsqueeze(0)
    output = inference(path_to_model_denoise = None,
                       path_to_model_sr = path_to_gen_model,
                       sr_model = Generator(),
                       denoise_model = None, image = lr_image_tensor)
    output = output.squeeze()
    save_image(output, path_to_image + '/sr_image.png')
    context.bot.send_photo(chat_id, open(path_to_image + '/sr_image.png', 'rb'))
    return RESULT


def get_denoised_image(update: Update, context: CallbackContext):
    """denoises image"""
    chat_id = update.message.chat_id
    file = context.bot.get_file(update.message.photo[-1].file_id)
    test_im = io.imread(file['file_path'])
    lr_image_tensor = transforms.functional.to_tensor(test_im).unsqueeze(0)
    output = inference(path_to_model_denoise = path_to_denoiser_model,
                       path_to_model_sr = None,
                       sr_model = None,
                       denoise_model = RidNet(), image = lr_image_tensor)
    output = output.squeeze()
    save_image(output, path_to_image + '/denoised_image.png')
    context.bot.send_photo(chat_id, open(path_to_image + '/denoised_image.png', 'rb'))
    return RESULT


def get_sr_denoise_img(update: Update, context: CallbackContext):
    """improves resolution and denoises image"""
    chat_id = update.message.chat_id
    file = context.bot.get_file(update.message.photo[-1].file_id)
    test_im = io.imread(file['file_path'])
    lr_image_tensor = transforms.functional.to_tensor(test_im).unsqueeze(0)
    output = inference(path_to_model_denoise = path_to_denoiser_model,
                       path_to_model_sr = path_to_gen_model,
                       sr_model = Generator(),
                       denoise_model = RidNet(), image = lr_image_tensor)
    output = output.squeeze()
    save_image(output, path_to_image + '/sr_denoised_image.png')
    context.bot.send_photo(chat_id, open(path_to_image + '/sr_denoised_image.png', 'rb'))
    return RESULT


def stop_bot(update: Update, context: CallbackContext):
    update.message.reply_text('Okey! See you later!')

    return END


def main() -> None:
    """Run the bot."""
    # Create the Updater and pass it your bot's token.
    updater = Updater("5031776824:AAFbKmZVwAr0Va-7OaH6G0tvssEf8PvmF-s")

    dispatcher = updater.dispatcher

    selection_handlers = [
        CallbackQueryHandler(super_resolution, pattern = '^' + str(IMG_RES) + '$'),
        CallbackQueryHandler(denoise, pattern = '^' + str(NOISE) + '$'),
        CallbackQueryHandler(sr_denoise, pattern = '^' + str(IMAGE_RES_NOISE) + '$'),
    ]

    start_handlers = [CommandHandler('start', start)]

    conv_handler = ConversationHandler(
        entry_points = [CommandHandler('start', start)],
        states = {
            SELECTING_ACTION: selection_handlers,
            IMG_FILE_SR: [MessageHandler(Filters.photo, get_SR_image)],
            IMG_FILE_DENOISE: [MessageHandler(Filters.photo, get_denoised_image)],
            IMG_FILE_SR_DENOISE: [MessageHandler(Filters.photo, get_sr_denoise_img)]
        },
        fallbacks = [CommandHandler('stop', stop_bot)]
    )
    dispatcher.add_handler(conv_handler)
    # Start the Bot
    updater.start_polling()

    # Run the bot until the user presses Ctrl-C or the process receives SIGINT,
    # SIGTERM or SIGABRT
    updater.idle()


if __name__ == '__main__':
    main()
