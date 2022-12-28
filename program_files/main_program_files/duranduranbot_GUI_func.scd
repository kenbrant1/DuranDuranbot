~duranduranbot_GUI_func = {

	if(w != nil, {w.close;});

	case{~fullscreenon != 1}{
		w = Window("DuranDuranbot",
			Rect(
				Window.availableBounds.width*(1/4.4),
				Window.availableBounds.height*(1/4.5),
				Window.availableBounds.width*((4.5-2)/4.5),
				Window.availableBounds.height*((4.5-2)/4.5)

			),

			/*Window.availableBounds.width*(0),
			Window.availableBounds.height*(0),
			Window.availableBounds.width,
			Window.availableBounds.height),*/

		/*0, 0, Window.screenBounds.width, Window.screenBounds.height,*/

			resizable: true, border:true);
	}
	{~fullscreenon == 1}{
		w = Window("DuranDuranbot",
			Rect(0, 0, Window.availableBounds.width, Window.availableBounds.height),

			resizable: true, border:true);
	};


	w.front; /*w.fullScreen;*/

	if(~randseed_1 == nil, {~randseed_1 =  99999999.rand;});
	if(~randseed_2 == nil, {~randseed_2 =  99999999.rand;});
	if(~randseed_3 == nil, {~randseed_3 =  99999999.rand;});

	if(~startprogram == nil, {~save_random_seeds_func.value; ~startprogram = 1;});

	if(~ai_assist_compose != 1, {AppClock.sched(1, {~main_page_func.value;});}, {
		if(~assist_loading_proj != 1, {~notepad_string = ~notepadview.string.asString;}, {~assist_is_loading = 0;});

		AppClock.sched(1, {~assist_compose_page_func.value;});
	});

	~hotkeys_func.value;

	/*AppClock.sched(4, {~firstview.close;
		~assist_compose_page_func.value;});*/

};