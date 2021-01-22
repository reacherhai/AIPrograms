;case 2
(define (problem rubik_problem)
    (:domain rubik)
    (:objects r g b o w y - color)
    (:init
        (color1 o g y)
        (color2 w b o)
        (color3 r b w)
        (color4 b o y)
        (color5 b r y)
        (color6 g o w)
        (color7 g w r)
        (color8 y r g)
    )
    (:goal
        (and
            (color1 b w o)
            (color2 b y o)
            (color3 b w r)
            (color4 b y r)
            (color5 g w o)
            (color6 g y o)
            (color7 g w r)
            (color8 g y r)
        )
    )

)