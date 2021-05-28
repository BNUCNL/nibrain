"""
    draw flag
"""

import pygame

def draw_flags(flag_name):

    # initiate pygame window
    pygame.init()
    screen = pygame.display.set_mode([600, 250])


    # draw flag
    if flag_name == 'france':
        print('****** France flag ******')
        screen.fill([255, 255, 255])
        flag_bluerect = pygame.draw.rect(screen, (3, 30, 159), (0, 0, 200, 250), 0)
        flag_whiterect = pygame.draw.rect(screen, (255, 255, 255), (200, 0, 200, 250), 0)
        flag_redrect = pygame.draw.rect(screen, (244, 42, 68), (400, 0, 200, 250), 0)

    if flag_name == 'japan':
        print('****** Japan flag ******')
        screen.fill([255, 255, 255])
        pygame.draw.circle(screen, [255, 0, 0], [300, 125], 60, 0)

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

    flag_name = 'japan'
    draw_flags(flag_name)
