(define (domain boxman_domain)
    (:requirements :strips :equality:typing )
    (:types  num loc )
    (:predicates 
                 (empty ?lcx -loc ?lcy -loc)
                 (person ?x -loc ?y - loc)
                 (box ?x - loc ?y - loc )
                 (inc ?x - loc ?y - loc )
    )


(:action moveup
    :parameters (?fromx - loc ?y -loc ?tox -loc)
    :precondition (and(person ?fromx ?y) (empty ?tox ?y) (inc ?tox ?fromx) )
    :effect (and (not (person ?fromx ?y)) (not(empty ?tox ?y)) (person ?tox ?y) (empty ?fromx ?y)  )
)

(:action pushup
    :parameters (?locP - loc ?y - loc ?locB - loc ?locD -loc )
    :precondition 
    (
        and 
        (person ?locP ?y)
        (box ?locB ?y)
        (empty ?locD ?y)
        (inc ?locB ?locP)
        (inc ?locD ?locB)
    )
    :effect 
    (
        and
        ( not (person ?locP ?y) )
        ( not (box ?locB ?y) )
        ( not (empty ?locD ?y ) )
        (empty ?locP ?y)
        (person ?locB ?y)
        (box ?locD ?y)
    )
)

(:action movedown
    :parameters (?fromx - loc ?y -loc ?tox -loc)
    :precondition (and(person ?fromx ?y) (empty ?tox ?y) (inc ?fromx ?tox) )
    :effect (and (not (person ?fromx ?y)) (not(empty ?tox ?y)) (person ?tox ?y) (empty ?fromx ?y)  )
)

(:action pushdown
    :parameters (?locP - loc ?y - loc ?locB - loc ?locD -loc )
    :precondition 
    (
        and 
        (person ?locP ?y)
        (box ?locB ?y)
        (empty ?locD ?y)
        (inc ?locP ?locB)
        (inc ?locB ?locD)
    )
    :effect 
    (
        and
        ( not (person ?locP ?y) )
        ( not (box ?locB ?y) )
        ( not (empty ?locD ?y ) )
        (empty ?locP ?y)
        (person ?locB ?y)
        (box ?locD ?y)
    )
)

(:action moveleft
    :parameters (?x - loc ?fromy -loc ?toy -loc)
    :precondition (and(person ?x ?fromy) (empty ?x ?toy) (inc ?toy ?fromy) )
    :effect (and (not (person ?x ?fromy)) (not(empty ?x ?toy)) (person ?x ?toy) (empty ?x ?fromy)  )
)

(:action pushleft
    :parameters (?x - loc ?locP - loc ?locB - loc ?locD -loc )
    :precondition 
    (
        and 
        (person ?x ?locP)
        (box ?x ?locB)
        (empty ?x ?locD)
        (inc ?locB ?locP)
        (inc ?locD ?locB)
    )
    :effect 
    (
        and
        ( not (person ?x ?locP) )
        ( not (box ?x ?locB) )
        ( not (empty ?x ?locB ) )
        (empty ?x ?locP)
        (person ?x ?locB)
        (box ?x ?locD)
    )
)

(:action moveright
    :parameters (?x - loc ?fromy -loc ?toy -loc)
    :precondition (and(person ?x ?fromy) (empty ?x ?toy) (inc ?fromy ?toy) )
    :effect (and (not (person ?x ?fromy)) (not(empty ?x ?toy)) (person ?x ?toy) (empty ?x ?fromy)  )
)

(:action pushright
    :parameters (?locP - loc ?x - loc ?locB - loc ?locD -loc )
    :precondition 
    (
        and 
        (person ?x ?locP)
        (box ?x ?locB)
        (inc ?locP ?locB)
        (empty ?x ?locD)
        (inc ?locB ?locD)
    )
    :effect 
    (
        and
        ( not (person ?x ?locP) )
        ( not (box ?x ?locB) )
        ( not (empty ?x ?locB ) )
        (empty ?x ?locP)
        (person ?x ?locB)
        (box ?x ?locD)
    )
    
)


)
