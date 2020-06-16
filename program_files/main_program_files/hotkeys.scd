// HOTKEYS

~hotkeys_func = {

	~spacebar_unicode = 32; //pause DDbot
	~esc_unicode = 27; //escape fullscreen

	w.view.keyDownAction = { arg view, char, modifiers, unicode, keycode;
		[char, modifiers, unicode, keycode];

		//pause/unpause synth
		if(((unicode == ~spacebar_unicode) or: (keycode == ~spacebar_unicode)), {
			case
			{~duranduranbot_routine.isPlaying == true}{~playbutton.valueAction = 0}
			{~duranduranbot_routine.isPlaying == false}{~playbutton.valueAction = 1};
		});
		if(((unicode == ~esc_unicode) or: (keycode == ~esc_unicode)), {
			case
			{~fullscreenon == 0}{~fullscreen_func.value;}
			{~fullscreenon == 1}{~end_fullscreen_func.value;};
		});
	};
};

	//To find key codes in post window... Hightlight the function and its parenthesis (without the /* and */) in red below,
	//and press "ctrl - enter" or "cmd - return" - then a small window called "test" will pop up in the taskbar...
	//Press the "type" button inside of the window, then press anywhere on the keyboard to find
	//the key codes in the right-hand side post window. Layout: [unicode, keycode, modifiers]. It is either a unicode
	//or a keycode. Do not permanently delete the /* and */ characters, or else the
	//program will not run.

	/*

	(
	var win = Window(\test, Rect(10, 100, 100, 50)),
	btn = Button(win, Rect(5, 5, 90, 40)).states_([["type"]]);
	btn.keyDownAction =
	{ arg view, char, modifiers, unicode, keycode;
	["unicode:"+unicode, "keycode:"+keycode, "modifiers:"+modifiers].postln;
	};
	win.front;
	)

	*/