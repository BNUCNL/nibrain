"""
    draw flag
"""

import pygame, sys

def draw_flags(flag_name):

    # initiate pygame window
    pygame.init()
    sceen = pygame.display.set_mode([400, 250])


    # draw flag
    if flag_name == 'france':
        print('****** France flag ******')
        sceen.fill([255, 255, 255])
        pygame.draw.polygon(sceen, [0, 0, 255], [0, 125], )

    if flag_name == 'japan':
        print('****** Japan flag ******')
        sceen.fill([255, 255, 255])
        pygame.draw.circle(sceen, [255, 0, 0], [200, 125], 60, 0)
        pygame.display.flip()

    # running pygame window
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
    pygame.quit()



if __name__ == '__main__':
    flag_name = 'france'
    draw_flags(flag_name)
