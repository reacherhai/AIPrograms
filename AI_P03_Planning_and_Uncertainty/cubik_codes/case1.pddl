﻿(define (problem rubik_problem)
    (:domain rubik)
    (:objects r g b o w y - color)
    (:init
        (color1 y o g)
        (color2 r b w)
        (color3 o b y)
        (color4 r g w)
        (color5 w o g)
        (color6 r y g)
        (color7 y r b)
        (color8 b o w)
    )
    (:goal
        (and
            (color1 w r g)
            (color2 w o g)
            (color3 w r b)
            (color4 w o b)
            (color5 y r g)
            (color6 y o g)
            (color7 y r b)
            (color8 y o b)
        )
    )

)
