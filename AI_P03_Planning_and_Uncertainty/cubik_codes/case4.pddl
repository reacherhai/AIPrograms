;case 4
(define (problem rubik_problem)
    (:domain rubik)
    (:objects r g b o w y - color)
    (:init
        (color1 y g r)
        (color2 o y g)
        (color3 o b y)
        (color4 o w g)
        (color5 y b r)
        (color6 b r w)
        (color7 o b w)
        (color8 g r w)
    )
    (:goal
        (and
            (color1 r b y)
            (color2 r g y)
            (color3 r b w)
            (color4 r g w)
            (color5 o b y)
            (color6 o g y)
            (color7 o b w)
            (color8 o g w)
        )
    )

)
