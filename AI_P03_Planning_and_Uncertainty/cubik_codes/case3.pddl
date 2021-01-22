;case 3
(define (problem rubik_problem)
    (:domain rubik)
    (:objects r g b o w y - color)
    (:init
        (color1 w o b)
        (color2 o y g)
        (color3 y r g)
        (color4 g o w)
        (color5 r w g)
        (color6 y r b)
        (color7 o y b)
        (color8 b w r)
    )
    (:goal
        (and
            (color1 r y g)
            (color2 r w g)
            (color3 r y b)
            (color4 r w b)
            (color5 o y g)
            (color6 o w g)
            (color7 o y b)
            (color8 o w b)
        )
    )

)
