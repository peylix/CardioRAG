import logging
def LoggingSave(feedback_data):
    logging.basicConfig(filename='feedback.log', level=logging.INFO,
                        format='%(asctime)s:%(levelname)s:%(message)s')

    for key, value in feedback_data.items():
        logging.info(f'{key}: {value}')
    
    logging.info('------------------------------------------------')