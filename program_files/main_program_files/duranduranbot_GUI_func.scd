~duranduranbot_GUI_func = {

	w = Window("DuranDuranbot",
		Rect(Window.availableBounds.width*(1/4.4), Window.availableBounds.height*(1/4.5), Window.availableBounds.width*((4.5-2)/4.5), Window.availableBounds.height*((4.5-2)/4.5)),

		/*Rect(0, 0, Window.screenBounds.width, Window.screenBounds.height),*/
		resizable: true, border:true);

	w.front; /*w.fullScreen;*/

	if(~randseed_1 == nil, {~randseed_1 =  99999999.rand;});
	if(~randseed_2 == nil, {~randseed_2 =  99999999.rand;});
	if(~randseed_3 == nil, {~randseed_3 =  99999999.rand;});

	if(~startprogram == nil, {~save_random_seeds_func.value; ~startprogram = 1;});

	if(~ai_assist_compose != 1, {~main_page_func.value;}, {~assist_compose_page_func.value;});

	if(~fullscreenon == 1, {~fullscreenbutton.valueAction = 1;});

	~hotkeys_func.value;

};