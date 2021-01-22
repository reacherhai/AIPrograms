can( move(Block,From,To), [clear(Block),clear(To),on(Block,From)] ):-
	block(Block),
	object(To),
	To \== Block,
	object(From),
	From \== To,
	Block \== From.
	
adds(move(X,From,To), [on(X,To),clear(From)]).
deletes(move(X,From,To), [on(X,From),clear(To)]).

object(X):- place(X); block(X).

plan(State,Goals, Plan, State):-
	satisfied(State, Goals). % plan empty

/*this is a simple planner*/
plan(State, Goals, Plan, FinalState):-
	conc(Plan,_,_),   /*short plans first*/
	conc(PrePlan,[Action | PostPlan],Plan), /*divide plan into */
	select(State,Goals,Goal), /*select a goal*/
	achieves(Action, Goal), /* relevant action*/
	can(Action, Condition), /* precondition */
	plan(State,Condition,PrePlan, MidState1), /*Enable Action*/
	apply(MidState1,Action, MidState2), /*Apply Action*/
	plan(MidState2,Goals,PostPlan,FinalState). /*Achieve remaining goals*/

conc([],L,L).
conc([H|L1],L2,[H|L3]):-
	conc(L1,L2,L3).

satisfied(State,[]).

satisfied(State, [Goal| Goals]):-
	member(Goal, State),
	satisfied(State,Goals).

select(State,Goals,Goal):-
	member(Goal,Goals),
	not( member(Goal,State) ). /*Goal not satisfied yet*/

achieves(Action, Goal):-
	adds(Action,Goals),
	member(Goal,Goals).
	
apply(State, Action, NewState):-
	deletes(Action, DelList),
	delete_all(State, DelList,State1),!,
	adds(Action, AddList),
	conc(AddList, State1, NewState).
	
preserves(Action,Goals):-
	deletes(Action, Relations),
	\+ (member(Goal,Relations),
		member(Goal,Goals)).

regress(Goals, Action, RegressedGoals):-
	adds(Action, AddRelations),
	delete_all(Goals,AddRelations,RestGoals),
	can(Action,Condition),
	addnew(Condition,RestGoals,RegressedGoals).

delete_all([],_,[]).

delete_all([X|L1],L2,Diff ):-   /*Diff is set-diff of L1 and L2*/
	member(X,L2),!,
	delete_all(L1,L2,Diff).

delete_all([X|L1],L2,[X|Diff]):-
	delete_all(L1,L2,Diff).
	
satisfied(State, Goals):- 
	delete_all(Goals, State, []).

/*addnew(NewGoals,Oldgoals,AllGoals)*/

addnew([],L,L).

addnew([Goal|_],Goals,_):-
	impossible(Goal,Goals),
	!,
	fail.

addnew([X|L1],L2,L3):-
	member(X,L2), !,
	addnew(L1,L2,L3).
	
addnew([X|L1],L2,[X|L3]):-
	addnew(L1,L2,L3).

/*impossible: incapitable*/
impossible(on(X,X),_).

impossible(on(X,Y),Goals):-
	member(clear(Y),Goals)
	;
	member(on(X,Y1),Goals), Y1\==Y
	;
	member(on(X1,Y),Goals), X1\==X.
	
impossible(clear(X),Goals):-
	member(on(_,X),Goals).
	
:- op(400,yfx,'#').

s(Goals#Sol#G, NewGoals#Soln#Gn, 1):-
	member(Goal,Goals),
	achieves(Action, Goal),
	can(Action, Condition),
	preserves(Action,Goals),
	regress(Goals,Action,NewGoals),
	Gn is G+1,
	conc([Action],Sol,Soln).
	

not_correct_position(A,State, Goals):-
	member(on(A,X),State),
	((member(on(A,Y),Goals),X\==Y);
	not_correct_position(X,State,Goals)).
	
/*thanks for TA's heuristic*/
h(Goals,H):-
	start(State),
	((setof(A,(block(A),not_correct_position(A,State,Goals)),Res),length(Res,H));
	(\+(setof(A,(block(A),not_correct_position(A,State,Goals)),Res)),H is 0)).

f(Goals#Sol#G, F):-
	h(Goals,H),
	F is G+H.
	
goal(Goals):-
	start(State),
	satisfied(State,Goals).


idastar(Start, Solution):-
	retract(next_bound(_)),
	fail
	;
	asserta(next_bound(0)),
	idastar0(Start#[]#0,Solution).

idastar0(Start, Sol):-
	retract(next_bound(B)),
	asserta(next_bound(99999)),
	f(Start,F),
	dfs([Start],F,B,Sol)
	;
	next_bound(NextB),
	NextB<99999,
	writeln(NextB),
	idastar0(Start,Sol).
	
dfs([N#Sol#G | Ns], F, B, Sol):-
    F=<B,
    goal(N).

dfs([N|Ns], F, B, Sol) :-
    F=<B,
    s(N,N1,_), \+(member(N1, Ns)),
    f(N1,F1),
    dfs([N1,N|Ns],F1,B,Sol).

dfs(_,F,B,_) :-
    F>B,
    update_bound(F),
    fail.
	
update_bound(F) :-
    next_bound(B),
    B=<F, !
    ;
    retract(next_bound(B)), !,
    asserta(next_bound(F)).






















