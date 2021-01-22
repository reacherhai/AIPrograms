
(define (domain rubik)

(:requirements :strips :typing :equality)
(:types color)

(:predicates
    (color1 ?f ?l ?u - color)
    (color2 ?f ?r ?u - color)
    (color3 ?f ?l ?d - color)
    (color4 ?f ?r ?d - color)
    (color5 ?b ?l ?u - color)
    (color6 ?b ?r ?u - color)
    (color7 ?b ?l ?d - color)
    (color8 ?b ?r ?d - color) 
)

;rotate the rubic's front size clockwise
(:action F
    :effect (and
        (forall (?f1 ?l1 ?u1 - color) ( when (color1 ?f1 ?l1 ?u1)
                (and
                    (not(color1 ?f1 ?l1 ?u1) )
                    (color2 ?f1 ?u1 ?l1)
                )
            )
        )
        
        (forall (?f2 ?r2 ?u2 - color) (when (color2 ?f2 ?r2 ?u2)
                (and
                    (not (color2 ?f2 ?r2 ?u2))
                    (color4 ?f2 ?u2 ?r2)
                )   
            )
        )
        
        (forall (?f3 ?l3 ?d3 - color)(when (color3 ?f3 ?l3 ?d3)
                (and
                    (not (color3 ?f3 ?l3 ?d3))
                    (color1 ?f3 ?d3 ?l3)
                )
            )
        )
        
        (forall (?f4 ?r4 ?d4 - color)(when (color4 ?f4 ?r4 ?d4)
                (and
                    (not (color4 ?f4 ?r4 ?d4))
                    (color3 ?f4 ?d4 ?r4)
                )    
            )
        )
    )
)

;rotate the rubic's front size counterclockwise
(:action F_p
    :effect (and
        (forall (?f1 ?l1 ?u1 - color) ( when (color1 ?f1 ?l1 ?u1)
                (and
                    (not(color1 ?f1 ?l1 ?u1) )
                    (color3 ?f1 ?u1 ?l1)
                )
            )
        )
        
        (forall (?f2 ?r2 ?u2 - color) (when (color2 ?f2 ?r2 ?u2)
                (and
                    (not (color2 ?f2 ?r2 ?u2))
                    (color1 ?f2 ?u2 ?r2)
                )   
            )
        )
        
        (forall (?f3 ?l3 ?d3 - color)(when (color3 ?f3 ?l3 ?d3)
                (and
                    (not (color3 ?f3 ?l3 ?d3))
                    (color4 ?f3 ?d3 ?l3)
                )
            )
        )
        
        (forall (?f4 ?r4 ?d4 - color)(when (color4 ?f4 ?r4 ?d4)
                (and
                    (not (color4 ?f4 ?r4 ?d4))
                    (color2 ?f4 ?d4 ?r4)
                )    
            )
        )
    )
)


; rotate the rubik's right side clockwise
(:action R
    :effect (and
        (forall (?f2 ?r2 ?u2 - color) ( when (color2 ?f2 ?r2 ?u2)
                (and
                    (not(color2 ?f2 ?r2 ?u2) )
                    (color6 ?u2 ?r2 ?f2)
                )
            )
        )
        
        (forall (?f4 ?r4 ?d4 - color) (when (color4 ?f4 ?r4 ?d4)
                (and
                    (not (color4 ?f4 ?r4 ?d4))
                    (color2 ?d4 ?r4 ?f4)
                )   
            )
        )
        
        (forall (?b6 ?r6 ?u6 - color)(when (color6 ?b6 ?r6 ?u6)
                (and
                    (not (color6 ?b6 ?r6 ?u6))
                    (color8 ?u6 ?r6 ?b6)
                )
            )
        )
        
        (forall (?b8 ?r8 ?d8 - color)(when (color8 ?b8 ?r8 ?d8)
                (and
                    (not (color8 ?b8 ?r8 ?d8))
                    (color4 ?d8 ?r8 ?b8)
                )    
            )
        )
    )
)

; rotate the rubik's right side counterclockwise
(:action R_p
    :effect (and
        (forall (?f2 ?r2 ?u2 - color) ( when (color2 ?f2 ?r2 ?u2)
                (and
                    (not(color2 ?f2 ?r2 ?u2) )
                    (color4 ?u2 ?r2 ?f2)
                )
            )
        )
        
        (forall (?f4 ?r4 ?d4 - color) (when (color4 ?f4 ?r4 ?d4)
                (and
                    (not (color4 ?f4 ?r4 ?d4))
                    (color8 ?d4 ?r4 ?f4)
                )   
            )
        )
        
        (forall (?b6 ?r6 ?u6 - color)(when (color6 ?b6 ?r6 ?u6)
                (and
                    (not (color6 ?b6 ?r6 ?u6))
                    (color2 ?u6 ?r6 ?b6)
                )
            )
        )
        
        (forall (?b8 ?r8 ?d8 - color)(when (color8 ?b8 ?r8 ?d8)
                (and
                    (not (color8 ?b8 ?r8 ?d8))
                    (color6 ?d8 ?r8 ?b8)
                )    
            )
        )
    )
)

; rotate the rubik's upper side clockwise
(:action U
    :effect (and
        (forall (?f1 ?l1 ?u1 - color) ( when (color1 ?f1 ?l1 ?u1)
                (and
                    (not(color1 ?f1 ?l1 ?u1) )
                    (color5 ?l1 ?f1 ?u1)
                )
            )
        )
        
        (forall (?f2 ?r2 ?u2 - color) (when (color2 ?f2 ?r2 ?u2)
                (and
                    (not (color2 ?f2 ?r2 ?u2))
                    (color1 ?r2 ?f2 ?u2)
                )   
            )
        )
        
        (forall (?b5 ?l5 ?u5 - color)(when (color5 ?b5 ?l5 ?u5)
                (and
                    (not (color5 ?b5 ?l5 ?u5))
                    (color6 ?l5 ?b5 ?u5)
                )
            )
        )
        
        (forall (?b6 ?r6 ?u6 - color)(when (color6 ?b6 ?r6 ?u6)
                (and
                    (not (color6 ?b6 ?r6 ?u6))
                    (color2 ?r6 ?b6 ?u6)
                )    
            )
        )
    )
)

; rotate the rubik's upper side counterclockwise
(:action U_p
    :effect (and
        (forall (?f1 ?l1 ?u1 - color) ( when (color1 ?f1 ?l1 ?u1)
                (and
                    (not(color1 ?f1 ?l1 ?u1) )
                    (color2 ?l1 ?f1 ?u1)
                )
            )
        )
        
        (forall (?f2 ?r2 ?u2 - color) (when (color2 ?f2 ?r2 ?u2)
                (and
                    (not (color2 ?f2 ?r2 ?u2))
                    (color6 ?r2 ?f2 ?u2)
                )   
            )
        )
        
        (forall (?b5 ?l5 ?u5 - color)(when (color5 ?b5 ?l5 ?u5)
                (and
                    (not (color5 ?b5 ?l5 ?u5))
                    (color1 ?l5 ?b5 ?u5)
                )
            )
        )
        
        (forall (?b6 ?r6 ?u6 - color)(when (color6 ?b6 ?r6 ?u6)
                (and
                    (not (color6 ?b6 ?r6 ?u6))
                    (color5 ?r6 ?b6 ?u6)
                )    
            )
        )
    )
)



)