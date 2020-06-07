~ai_assist_settings_page_func = {

	if(~firstview != nil, {~firstview.close;});
	if(~firstview2 != nil, {~firstview2.close;});

	~firstview = View(w, Rect(0, 0, w.bounds.width, w.bounds.height));
	/*view.clearOnRefresh = false;*/
	~firstview.background = ~backgroundcolor;
	~firstviewboundsWIDTH = ~firstview.bounds.width;
	~firstviewboundsHEIGHT = ~firstview.bounds.height;

	~firstviewtext1 = StaticText(~firstview, Rect(~firstviewboundsWIDTH/2.2857142857143, ~firstviewboundsHEIGHT/43.7, ~firstviewboundsWIDTH/8, ~firstviewboundsHEIGHT/21));
	~firstviewtext1.background=~backgroundcolor;
	~firstviewtext1.stringColor_(~stringcolor);
	~firstviewtext1.string = "Assist Settings";
	~firstviewtext1.align = \topLeft;
	~firstviewtext1.font = Font(size: ~firstviewboundsWIDTH/48);

	~firstviewtext2 = Button.new(~firstview, Rect(~firstviewboundsWIDTH/72, ~firstviewboundsHEIGHT/43.7, ~firstviewboundsWIDTH/16, ~firstviewboundsHEIGHT/21)).states_([["back",~stringcolor,~backgroundcolor],["back",~stringcolor,~backgroundcolor]]).action_({arg button;
		~firstview.close;
		~assist_compose_page_func.value;
	});
	~firstviewtext2.background=~stringcolor;
	~firstviewtext2.string = "Back";
	~firstviewtext2.font = Font(size: ~firstviewboundsWIDTH/48);

	~ml_off_on_button = Button.new(~firstview, Rect(~firstviewboundsWIDTH/72, ~firstviewboundsHEIGHT/7.5, ~firstviewboundsWIDTH/3.5, ~firstviewboundsHEIGHT/21)).states_([["Turn Machine Learning OFF",~stringcolor,~backgroundcolor],["Turn Machine Learning ON",~stringcolor,~backgroundcolor]]).action_({arg button;

		~mlfullpieceprompt = View(~firstview, Rect((~firstviewboundsWIDTH/3), ~firstviewboundsHEIGHT/25, ~firstviewboundsWIDTH/3, ~firstviewboundsHEIGHT/3));
		~mlfullpieceprompt.background_(~backgroundcolor);

		~mlfullpieceprompttext = StaticText(~mlfullpieceprompt, Rect(0, 0, ~mlfullpieceprompt.bounds.width, ~mlfullpieceprompt.bounds.height/3));
		~mlfullpieceprompttext.background_(~backgroundcolor);
		~mlfullpieceprompttext.stringColor_(~stringcolor);
		~mlfullpieceprompttext.align = \center;

		if(~assist_ml_on == 0, {
			~mlfullpieceprompttext.string = "Turn machine learning on?";

		}, {
			~mlfullpieceprompttext.string = "Turn machine learning off?";

		});
		~mlfullpieceprompttext.font = Font(size: ~mlfullpieceprompt.bounds.width/14);
		~mlfullpieceprompttext.front;

		~train_full_piece_yes_no_button_1 = Button.new(~mlfullpieceprompt, Rect(~mlfullpieceprompt.bounds.width/8, ~mlfullpieceprompt.bounds.height/1.6, ~mlfullpieceprompt.bounds.width/4, ~mlfullpieceprompt.bounds.height/5)).states_([["Yes",~stringcolor,~backgroundcolor]]).action_({

			if(~assist_ml_on == 0, {
				~assist_ml_on = 1;
				g = File((~ddbot_root)++"/program_files/other_program_files/assist_ml_on.scd", "w");
				g.write(~assist_ml_on.asString);

			}, {
				~assist_ml_on = 0;
				g = File((~ddbot_root)++"/program_files/other_program_files/assist_ml_on.scd", "w");
				g.write(~assist_ml_on.asString);

			});
			~mlfullpieceprompt.close;
			if(~assist_ml_on == 1, {~ml_off_on_button.value = 0;}, {~ml_off_on_button.value = 1;});
		});
		~train_full_piece_yes_no_button_1.font = Font(size: ~firstviewboundsWIDTH/48);

		~train_full_piece_yes_no_button_2 = Button.new(~mlfullpieceprompt, Rect( ~mlfullpieceprompt.bounds.width/1.6, ~mlfullpieceprompt.bounds.height/1.6, ~mlfullpieceprompt.bounds.width/4, ~mlfullpieceprompt.bounds.height/5)).states_([["No",~stringcolor,~backgroundcolor]]).action_({

			~mlfullpieceprompt.close;
			if(~assist_ml_on == 1, {~ml_off_on_button.value = 0;}, {~ml_off_on_button.value = 1;});
		});
		~train_full_piece_yes_no_button_2.font = Font(size: ~firstviewboundsWIDTH/48);

	});
	~ml_off_on_button.font = Font(size: ~firstviewboundsWIDTH/48);

	if(~assist_ml_on == 1, {~ml_off_on_button.value = 0;}, {~ml_off_on_button.value = 1;});

	~ml_data_off_on_button = Button.new(~firstview, Rect(~firstviewboundsWIDTH/72, ~firstviewboundsHEIGHT/4.7, ~firstviewboundsWIDTH/3.5, ~firstviewboundsHEIGHT/21)).states_([["Turn using ML data OFF",~stringcolor,~backgroundcolor],["Turn using ML data ON",~stringcolor,~backgroundcolor]]).action_({arg button;

		~mlfullpieceprompt = View(~firstview, Rect((~firstviewboundsWIDTH/3), ~firstviewboundsHEIGHT/25, ~firstviewboundsWIDTH/3, ~firstviewboundsHEIGHT/3));
		~mlfullpieceprompt.background_(~backgroundcolor);

		~mlfullpieceprompttext = StaticText(~mlfullpieceprompt, Rect(0, 0, ~mlfullpieceprompt.bounds.width, ~mlfullpieceprompt.bounds.height/3));
		~mlfullpieceprompttext.background_(~backgroundcolor);
		~mlfullpieceprompttext.stringColor_(~stringcolor);
		~mlfullpieceprompttext.align = \center;

		if(~assist_ml_data_on == 0, {
			~mlfullpieceprompttext.string = "Turn use ML data on?";

		}, {
			~mlfullpieceprompttext.string = "Turn use ML data off?";

		});
		~mlfullpieceprompttext.font = Font(size: ~mlfullpieceprompt.bounds.width/14);
		~mlfullpieceprompttext.front;

		~train_full_piece_yes_no_button_1 = Button.new(~mlfullpieceprompt, Rect(~mlfullpieceprompt.bounds.width/8, ~mlfullpieceprompt.bounds.height/1.6, ~mlfullpieceprompt.bounds.width/4, ~mlfullpieceprompt.bounds.height/5)).states_([["Yes",~stringcolor,~backgroundcolor]]).action_({

			if(~assist_ml_data_on == 0, {
				~assist_ml_data_on = 1;
				g = File((~ddbot_root)++"/program_files/other_program_files/assist_ml_data_on.scd", "w");
				g.write(~assist_ml_data_on.asString);

			}, {
				~assist_ml_data_on = 0;
				g = File((~ddbot_root)++"/program_files/other_program_files/assist_ml_data_on.scd", "w");
				g.write(~assist_ml_data_on.asString);

			});
			~mlfullpieceprompt.close;
			if(~assist_ml_data_on == 1, {~ml_data_off_on_button.value = 0;}, {~ml_data_off_on_button.value = 1;});
		});
		~train_full_piece_yes_no_button_1.font = Font(size: ~firstviewboundsWIDTH/48);

		~train_full_piece_yes_no_button_2 = Button.new(~mlfullpieceprompt, Rect( ~mlfullpieceprompt.bounds.width/1.6, ~mlfullpieceprompt.bounds.height/1.6, ~mlfullpieceprompt.bounds.width/4, ~mlfullpieceprompt.bounds.height/5)).states_([["No",~stringcolor,~backgroundcolor]]).action_({

			~mlfullpieceprompt.close;
			if(~assist_ml_data_on == 1, {~ml_data_off_on_button.value = 0;}, {~ml_data_off_on_button.value = 1;});
		});
		~train_full_piece_yes_no_button_2.font = Font(size: ~firstviewboundsWIDTH/48);

	});
	~ml_data_off_on_button.font = Font(size: ~firstviewboundsWIDTH/48);

	if(~assist_ml_data_on == 1, {~ml_data_off_on_button.value = 0;}, {~ml_data_off_on_button.value = 1;});

};