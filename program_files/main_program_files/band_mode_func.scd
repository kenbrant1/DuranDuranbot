~band_mode_func = {

if(~bandmodeview1 != nil, {~bandmodeview1.close;});

	~button_base_bandmode = {(~bandmodeview1HEIGHT/3.7+(((~bandmodeview1HEIGHT/3)-(~bandmodeview1HEIGHT/3.7))*x))};

	~bandmodeview1 = ScrollView(w, Rect(w.bounds.height*(1/4.4), w.bounds.width*(1/4.5), w.bounds.width*((4.5-2)/4.5), w.bounds.height*((4.5-2)/4.5)));
	/*view.clearOnRefresh = false;*/
	~bandmodeview1.background = ~backgroundcolor;
	~bandmodeview1WIDTH = w.bounds.width*((4.5-2)/4.5);
	~bandmodeview1HEIGHT = w.bounds.height*((4.5-2)/4.5);
	~bandmodeview1.front;

	~bandmodetext1 = StaticText(~bandmodeview1, Rect(0, 0, ~bandmodeview1WIDTH/1, ~bandmodeview1HEIGHT/12));
	~bandmodetext1.background=~backgroundcolor;
	~bandmodetext1.stringColor_(~stringcolor);
	~bandmodetext1.string = "Enter user names (press enter to submit) -";
	~bandmodetext1.align = \topLeft;
	~bandmodetext1.font = Font(~default_font, size: ~firstviewboundsWIDTH/48);

	if(~name_array == nil, {~name_array = [];});

	~bandmodetextfield = TextField(~bandmodeview1, Rect(0, ~firstviewboundsHEIGHT/12, ~firstviewboundsWIDTH/3, ~firstviewboundsHEIGHT/21));
	~bandmodetextfield.string = "enter here".asString;
	~bandmodetextfield.background_(~backgroundcolor);
	~bandmodetextfield.stringColor_(~stringcolor);
	~bandmodetextfield.font = Font(~default_font, size: ~firstviewboundsWIDTH/48);
	~bandmodetextfield.action = {arg name;
		~name_array = ~name_array.add(name.value.asString);
		~bandmodetext2.string = ~name_array.value.asString;
		~bandmodetextfield.string = "".asString;
	};

	~clear_users_button = Button.new(~bandmodeview1, Rect(0, ~firstviewboundsWIDTH/12, ~bandmodeview1WIDTH/3.8, ~bandmodeview1WIDTH/21)).states_([["Clear Users",~stringcolor,~backgroundcolor]]).action_({arg button;

		~name_array = [];
		~user_x = 0;
		~bandmodetext2.string = ~name_array.value.asString;
	});
	~clear_users_button.font = Font(~default_font, size: ~firstviewboundsWIDTH/48);

	~bandmodetext2 = StaticText(~bandmodeview1, Rect(0, ~firstviewboundsWIDTH/7, ~bandmodeview1WIDTH/0.01, ~bandmodeview1HEIGHT/12));
	~bandmodetext2.background=~backgroundcolor;
	~bandmodetext2.stringColor_(~stringcolor);
	~bandmodetext2.string = ~name_array.value.asString;
	~bandmodetext2.font = Font(~default_font, size: ~firstviewboundsWIDTH/48);

	~users_done_button = Button.new(~bandmodeview1, Rect(0, ~firstviewboundsWIDTH/5, ~bandmodeview1WIDTH/3.8, ~bandmodeview1WIDTH/21)).states_([["Done",~stringcolor,~backgroundcolor]]).action_({arg button;

		~bandmodetext3.string = ~name_array.at(~user_x).value.asString;
		~bandmodeview1.close;
	});
	~users_done_button.font = Font(~default_font, size: ~firstviewboundsWIDTH/48);


};