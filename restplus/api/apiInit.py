import logging
import traceback
from flask_restplus import Api
from restplus import settings

log = logging.getLogger(__name__)

api = Api(version='1.0', title='Handwritten Digit Recognition API',
          description= "An API to upload an image of a handwritten digit to \
          obtain pobabilty of it being a digit from 0 to 9.")

@api.errorhandler
def default_error_handler(e):
    message = 'An unhandled exception occurred.'
    log.exception(message)

    if not settings.FLASK_DEBUG:
        return {'message': message}, 500