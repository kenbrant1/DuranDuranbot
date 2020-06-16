~assist_slider_lenth_func = {

	~timeslider.controlSpec = ControlSpec(0, ("assist_inst_"++(1)++"_array_part_"++(~assist_part_number)).asSymbol.envirGet.size - 1, \lin, 1, 0, "bits");
	~cutslider1.controlSpec = ControlSpec(0, ("assist_inst_"++(1)++"_array_part_"++(~assist_part_number)).asSymbol.envirGet.size - 1, \lin, 1, 0, "bits");
	~cutslider2.controlSpec = ControlSpec(0, ("assist_inst_"++(1)++"_array_part_"++(~assist_part_number)).asSymbol.envirGet.size - 1, \lin, 1, 0, "bits");
};

~assist_compose_page_func = {var fillertext,fillertext2;

	/*
	WHAT WE WANT

	- All functionality on the assist compose page - play button, rate, slider, etc
	- *everything* can be saved for a later date
	-
	*/

	~ai_assist_compose = 1;

	~avg_part_length = ~assist_avg_part_length;
	~avg_part_range = ~assist_avg_part_range;

	~hotkeys_func.value;

	~button_base = {(~firstviewboundsHEIGHT/3.7+(((~firstviewboundsHEIGHT/3)-(~firstviewboundsHEIGHT/3.7))*x))};

	/*~firstview = ScrollView(w, Rect(0, 0, w.bounds.width, w.bounds.height));
	/*view.clearOnRefresh = false;*/
	~firstview.background = ~backgroundcolor;
	~firstviewboundsWIDTH = ~firstview.bounds.width;
	~firstviewboundsHEIGHT = ~firstview.bounds.height;

	~secondview = ScrollView(~firstview, Rect(0, 200, w.bounds.width, w.bounds.height));
	/*view.clearOnRefresh = false;*/
	~firstview.background = ~backgroundcolor;
	~firstviewboundsWIDTH = ~firstview.bounds.width;
	~firstviewboundsHEIGHT = ~firstview.bounds.height;*/

	/*~firstview = ScrollView(w, Rect(0, 0, w.bounds.width, w.bounds.height));
	/*view.clearOnRefresh = false;*/
	~firstview.background = ~backgroundcolor;*/

	~firstview = ScrollView(w, Rect(0, 0, w.bounds.width, w.bounds.height/2));
	/*view.clearOnRefresh = false;*/
	~firstview.background = ~backgroundcolor;
	~firstviewboundsWIDTH = ~firstview.bounds.width;
	~firstviewboundsHEIGHT = ~firstview.bounds.height*2;

	~firstview2 = ScrollView(w, Rect(0, w.bounds.height/2, w.bounds.width, w.bounds.height - (w.bounds.height/2)));
	/*view.clearOnRefresh = false;*/
	~firstview2.background = ~backgroundcolor;
	~firstviewboundsWIDTH2 = ~firstview.bounds.width;
	~firstviewboundsHEIGHT2 = ~firstview.bounds.height*2;

	~firstviewtext1 = StaticText(~firstview, Rect(~firstviewboundsWIDTH/2.75, ~firstviewboundsHEIGHT/43.7, ~firstviewboundsWIDTH/4, ~firstviewboundsHEIGHT/21));
	~firstviewtext1.background=~backgroundcolor;
	~firstviewtext1.stringColor_(~stringcolor);
	~firstviewtext1.string = "DDbot Assist Composition";
	~firstviewtext1.align = \topLeft;
	~firstviewtext1.font = Font(~default_font, size: ~firstviewboundsWIDTH/48);

	~firstviewtext11 = StaticText(~firstview, Rect(~firstviewboundsWIDTH/2.75, ~firstviewboundsHEIGHT/43.7+(~firstviewboundsHEIGHT/21), ~firstviewboundsWIDTH/4, ~firstviewboundsHEIGHT/21));
	~firstviewtext11.background=~backgroundcolor;
	~firstviewtext11.stringColor_(~stringcolor);
	~firstviewtext11.string = "(scroll down to see more)";
	~firstviewtext11.font = Font(~default_font, size: ~firstviewboundsWIDTH/55);

	~firstviewtext2 = Button.new(~firstview, Rect(~firstviewboundsWIDTH/72, ~firstviewboundsHEIGHT/43.7, ~firstviewboundsWIDTH/16, ~firstviewboundsHEIGHT/21)).states_([["back",~stringcolor,~backgroundcolor],["back",~stringcolor,~backgroundcolor]]).action_({arg button;
		~firstview.close;
		~main_page_func.value;
	});
	~firstviewtext2.background=~stringcolor;
	~firstviewtext2.string = "Back";
	~firstviewtext2.font = Font(~default_font, size: ~firstviewboundsWIDTH/48);

	~generatepiecebutton = Button.new(~firstview, Rect(~firstviewboundsWIDTH/72, ~firstviewboundsHEIGHT/7.5, ~firstviewboundsWIDTH/6.54, ~firstviewboundsHEIGHT/21)).states_([["Play full piece",~stringcolor,~backgroundcolor]]).action_({arg button;
		var piecegenerated;
		/*~piecegenerated = StaticText(~firstview, Rect(~firstviewboundsWIDTH/72, ~firstviewboundsHEIGHT/9.5, ~firstviewboundsWIDTH/6.2608695652174, ~firstviewboundsWIDTH/36));
		~piecegenerated.background=~backgroundcolor;
		~piecegenerated.stringColor_(~stringcolor);
		"Please Wait".postln;
		~piecegenerated.font = Font(~default_font, size: ~firstviewboundsWIDTH/55);
		~piecegenerated.string = "Please Wait";/**/
		if(~piece_created == "true", {~playbutton.close; ~arrow.close;});*//*
		~piece_generator_func.value;*/

		~load_piece_to_main_page_button.valueAction = 0;
		if(("instrument_"++(1)++"_full").asSymbol.envirGet.flatten.at(0).isNumber == true, {
			if(~duranduranbot_routine.isPlaying == true, {~stop_func.value;});
			~load_assist = 1;
			~playbutton.valueAction = 1;
			~load_assist = 0;
			AppClock.sched(0, {~playbutton.value = 1;});
		});

	});
	~generatepiecebutton.font = Font(~default_font, size: ~firstviewboundsWIDTH/48);

	/*if(playnil == nil, {nil}, {*/
	/*~play_visible_func = {
	~play_visible_func_verify*/

	~playbutton_func.value;
	if(~pause_piece == 1, {~playbutton.string = "Resume"; ~stopbuttonfunc.value;});

	if(~piece_created == nil, {~playbutton.close; ~arrow.close;});

	~stopbuttonfunc = {

		~stopbutton = Button.new(~firstview, Rect(~firstviewboundsWIDTH/2.85, ~firstviewboundsHEIGHT/7.5, ~firstviewboundsWIDTH/8, ~firstviewboundsHEIGHT/21)).states_([["Stop piece",~stringcolor,~backgroundcolor]]).action_({arg button;
			~stop_func.value;
		});
		~stopbutton.font = Font(~default_font, size: ~firstviewboundsWIDTH/48);
	};

	if(~duranduranbot_routine.isPlaying == true, {~playbutton.value = 1; ~stopbutton.close; ~stopbuttonfunc.value}, {~playbutton.value = 0});

	if(~piece_position == nil, {~piece_position = 0});

	~timeslider_func.value;

	~firstviewtext3 = Button.new(~firstview, Rect(~firstviewboundsWIDTH/1.37, ~firstviewboundsHEIGHT/43.7, ~firstviewboundsWIDTH/3.8, ~firstviewboundsHEIGHT/21)).states_([["Instructions",~stringcolor,~backgroundcolor],["Instructions",~stringcolor,~backgroundcolor]]).action_({arg button;
		~ai_assist_instructions_page_func.value;

	});
	~firstviewtext3.font = Font(~default_font, size: ~firstviewboundsWIDTH/48);

	~save_assist_project_button = Button.new(~firstview, Rect(~firstviewboundsWIDTH/1.37, ~firstviewboundsHEIGHT/43.7+(~firstviewboundsHEIGHT/21), ~firstviewboundsWIDTH/3.8, ~firstviewboundsHEIGHT/21)).states_([["Save Project",~stringcolor,~backgroundcolor]]).action_({arg button;

		"please wait".postln;
		if(~set_trim_on != 1, {
			g = File((~ddbot_root)++"/program_files/saved_projects/project_"++(z=((Date.getDate.format("%d_%m_%Y__%H_%M_%S").asString)))++".scd", "w");
		}, {
			g = File((~ddbot_root)++"/program_files/saved_projects/project_"++(z=((Date.getDate.format("%d_%m_%Y__%H_%M_%S").asString)))++"_trim.scd", "w");
		});
		g.write(
			"~synth_choice_1 = "++(~synth_choice_1.asString.quote;)++
			";\n~synth_choice_2 = "++(~synth_choice_2.asString.quote;)++
			";\n~synth_choice_3 = "++(~synth_choice_3.asString.quote;)++
			";\n~composition_process_setting = "++(~composition_process_setting.asString)++
			";\n~number_of_instruments = "++(~number_of_instruments.asString)++
			";\n~bit_duration = "++(~bit_duration.reciprocal.asString;)++".reciprocal"++
			";\n~tuning_frequency = "++(~tuning_frequency.asString)++
			";\n~which_players = "++(~which_players.asString)++
			";\n~amp_1 = "++(~amp_1.asString)++
			";\n~amp_2 = "++(~amp_2.asString)++
			";\n~amp_3 = "++(~amp_3.asString)++
			";\n~num_assist_parts = "++(~num_assist_parts.asString)++
			";\n~line_multiple1 = "++(~line_multiple1.asString)++
			";\n~line_multiple2 = "++(~line_multiple2.asString)++";";

		);
		~num_assist_parts.do(j = 0; {
			j=j+1;
			~number_of_instruments.do(s = 0; {
				s=s+1;
				g.write("\n~assist_inst_"++(s)++"_array_part_"++(j)++" = [ ");
				("assist_inst_"++(s)++"_array_part_"++(j)).asSymbol.envirGet.size.do(q = (-1); {
					q=q+1;
					g.write((("assist_inst_"++(s)++"_array_part_"++(j)).asSymbol.envirGet.at(q).asString)++", ");
				});
				g.write("];");

			});
		});
		~num_assist_parts.do(j = 0; {
			j=j+1;
			~number_of_instruments.do(s = 0; {
				s=s+1;
				g.write("\n~assist_inst_"++(s)++"_comp_interval_array_part_"++(j)++" = [ ");
				("assist_inst_"++(s)++"_comp_interval_array_part_"++(j)).asSymbol.envirGet.size.do(q = (-1); {
					q=q+1;
					g.write((("assist_inst_"++(s)++"_comp_interval_array_part_"++(j)).asSymbol.envirGet.at(q).asString)++", ");
				});
				g.write("];");

			});
		});
		~num_assist_parts.do(j = 0; {
			j=j+1;
			~number_of_instruments.do(s = 0; {
				s=s+1;
				g.write("\n~assist_inst_"++(s)++"_comp_choice_array_part_"++(j)++" = [ ");
				("assist_inst_"++(s)++"_comp_choice_array_part_"++(j)).asSymbol.envirGet.size.do(q = (-1); {
					q=q+1;
					g.write((("assist_inst_"++(s)++"_comp_choice_array_part_"++(j)).asSymbol.envirGet.at(q).asString)++", ");
				});
				g.write("];");

			});
		});

		g.close;

		"project saved".postln;
	});
	~save_assist_project_button.font = Font(~default_font, size: ~firstviewboundsWIDTH/48);

	~assist_clear_all_func = {

		if(~duranduranbot_routine.isPlaying == true, {~stop_func.value;});

		~num_assist_parts.do(j = 0; {
			j=j+1;
			~number_of_instruments.do(s = 0; {
				s=s+1;
				("assist_inst_"++(s)++"_array_part_"++(j)).asSymbol.envirPut(nil);
				("assist_inst_"++(s)++"_array_part_"++(j)++"_midi").asSymbol.envirPut(nil);

				("assist_inst_"++(s)++"_comp_interval_array_part_"++(j)).asSymbol.envirPut(nil);

				("assist_inst_"++(s)++"_comp_choice_array_part_"++(j)).asSymbol.envirPut(nil);

			});
		});
		~num_assist_parts = nil;
		~line_multiple1 = nil;
		~line_multiple2 = nil;

	};

	~load_saved_assist_project_button = Button.new(~firstview, Rect(~firstviewboundsWIDTH/1.37, ~firstviewboundsHEIGHT/43.7+(~firstviewboundsHEIGHT/21*2), ~firstviewboundsWIDTH/3.8, ~firstviewboundsHEIGHT/21)).states_([["Load Saved Project",~stringcolor,~backgroundcolor]]).action_({arg button;

		FileDialog.new(
			{arg path;

				~assist_clear_all_func.value;
				File.readAllString(path, r).interpret.value;
				~num_assist_parts.do(j = 0; {
					j=j+1;
					~number_of_instruments.do(s = 0; {
						s=s+1;
						("assist_inst_"++(s)++"_array_part_"++(j)++"_midi").asSymbol.envirPut(
							("assist_inst_"++(s)++"_array_part_"++(j)).asSymbol.envirGet.cpsmidi;
						);
					});
				});
				w.close;
				g = File((~ddbot_root)++"/program_files/other_program_files/synth_choice_1.scd", "w");
				g.write(~synth_choice_1.asString.quote);
				g.close;
				g = File((~ddbot_root)++"/program_files/other_program_files/synth_choice_2.scd", "w");
				g.write(~synth_choice_2.asString.quote);
				g.close;
				g = File((~ddbot_root)++"/program_files/other_program_files/synth_choice_3.scd", "w");
				g.write(~synth_choice_3.asString.quote);
				g.close;
				g = File((~ddbot_root)++"/program_files/other_program_files/composition_process_setting.scd", "w");
				g.write(~composition_process_setting.asString);
				g.close;
				g = File((~ddbot_root)++"/program_files/other_program_files/numberofvoices.scd", "w");
				g.write(~number_of_instruments.asString);
				g.close;
				g = File((~ddbot_root)++"/program_files/other_program_files/bit_speed.scd", "w");
				g.write(~bit_duration.asString);
				g.close;
				g = File((~ddbot_root)++"/program_files/other_program_files/tuning_frequency.scd", "w");
				g.write(~tuning_frequency.asString);
				g.close;
				g = File((~ddbot_root)++"/program_files/other_program_files/which_players.scd", "w");
				g.write(~which_players.asString);
				g.close;
				g = File((~ddbot_root)++"/program_files/other_program_files/amplitude_1.scd", "w");
				g.write(~amp_1.asString);
				g.close;
				g = File((~ddbot_root)++"/program_files/other_program_files/amplitude_2.scd", "w");
				g.write(~amp_2.asString);
				g.close;
				g = File((~ddbot_root)++"/program_files/other_program_files/amplitude_3.scd", "w");
				g.write(~amp_3.asString);
				g.close;
				~duranduranbot_GUI_func.value;
				"project loaded".postln;
				if(~fullscreenon == 1, {~fullscreenbutton.valueAction = 1;});
				~assist_compose_page_func.value;
		}, {}, 0, 0, stripResult: true, path: ((~ddbot_root)++"/program_files/saved_projects"));
	});
	~load_saved_assist_project_button.font = Font(~default_font, size: ~firstviewboundsWIDTH/48);

	~new_assist_project_button = Button.new(~firstview, Rect(~firstviewboundsWIDTH/1.37, ~firstviewboundsHEIGHT/43.7+(~firstviewboundsHEIGHT/21*3), ~firstviewboundsWIDTH/3.8, ~firstviewboundsHEIGHT/21)).states_([["New Proj. (will save curr.)",~stringcolor,~backgroundcolor]]).action_({arg button;

		~save_assist_project_button.valueAction = 0;
		~assist_clear_all_func.value;
		w.close;
		~duranduranbot_GUI_func.value;
		"project loaded".postln;
		if(~fullscreenon == 1, {~fullscreenbutton.valueAction = 1;});
		~assist_compose_page_func.value;
	});
	~new_assist_project_button.font = Font(~default_font, size: ~firstviewboundsWIDTH/48);


	/*~spacing1 = {(~firstviewboundsWIDTH2/72) + (~firstviewboundsWIDTH2/18 * ~line_multiple1)};
	~spacing2 = {(~firstviewboundsHEIGHT2/21*~line_multiple2)};

	if(~assist_part_number == nil, {~assist_part_number = 1});

	~new_part_button = Button.new(~firstview, Rect(~firstviewboundsWIDTH/72, ~firstviewboundsHEIGHT/3.7, ~firstviewboundsWIDTH/9, ~firstviewboundsHEIGHT/21)).states_([["New Part",~stringcolor,~backgroundcolor]]).action_({arg button;

	~num_assist_parts = ~num_assist_parts + 1;
	if(~line_multiple2 >= 10, {~line_multiple1 = ~line_multiple1 + 1; ~line_multiple2 = 0;});
	~line_multiple2 = ~line_multiple2 + 1;

	("assist_comp_part_"++(~num_assist_parts)++"_button").asSymbol.envirPut(
	Button.new(~firstview2, Rect(~spacing1.value, ~spacing2.value, ~firstviewboundsWIDTH/18, ~firstviewboundsHEIGHT/21)).states_([[~num_assist_parts.asString,~stringcolor,~backgroundcolor],[~num_assist_parts.asString,Color.black,Color.green]]).action_({arg button;
	("assist_comp_part_"++(~assist_part_number)++"_button").asSymbol.envirGet.value = 0;
	button.value = 1;
	~assist_part_number = button.string.asInteger;
	if(~duranduranbot_routine.isPlaying == true, {~stop_func.value;});
	AppClock.sched(0.25, {~playbutton.valueAction = 1;});
	});
	);
	("assist_comp_part_"++(~num_assist_parts)++"_button").asSymbol.envirGet.font = Font(~default_font, size: ~firstviewboundsWIDTH/48);

	});
	~new_part_button.font = Font(~default_font, size: ~firstviewboundsWIDTH/48);*/

	~spacing1 = {(~firstviewboundsWIDTH2/18 * ~line_multiple2)};
	~spacing2 = {(~firstviewboundsHEIGHT2/21 * ~line_multiple1)};

	if(~assist_part_number == nil, {~assist_part_number = 1});

	~new_part_button = Button.new(~firstview, Rect(~firstviewboundsWIDTH/72, (x=0; ~button_base.value;), ~firstviewboundsWIDTH/9, ~firstviewboundsHEIGHT/21)).states_([["New Part",~stringcolor,~backgroundcolor]]).action_({arg button;

		~num_assist_parts = ~num_assist_parts + 1;
		if(~line_multiple2 >= 16, {~line_multiple1 = ~line_multiple1 + 1; ~line_multiple2 = 0;});
		~line_multiple2 = ~line_multiple2 + 1;

		("assist_comp_part_"++(~num_assist_parts)++"_button").asSymbol.envirPut(
			Button.new(~firstview2, Rect(~spacing1.value, ~spacing2.value, ~firstviewboundsWIDTH/18, ~firstviewboundsHEIGHT/21)).states_([[~num_assist_parts.asString,~stringcolor,~backgroundcolor],[~num_assist_parts.asString,Color.black,Color.green]]).action_({arg button;
				("assist_comp_part_"++(~assist_part_number)++"_button").asSymbol.envirGet.value = 0;
				button.value = 1;
				~assist_part_number = button.string.asInteger;
				if(~duplicate_on == 1, {
					if(~dup_1_part_num == nil, {~dup_1_part_num = ~assist_part_number}, {
						("assist_comp_part_"++(~dup_1_part_num)++"_button").asSymbol.envirGet.value = 1;
						~dup_2_part_num = ~assist_part_number;
					});
				});
				if(~delete_on == 1, {
					~delete_array = ~delete_array.add(~assist_part_number);
					~delete_array.size.do(j=(-1); {
						j=j+1;
						("assist_comp_part_"++(~delete_array.at(j))++"_button").asSymbol.envirGet.value = 1;
					});
				});
				if(~duranduranbot_routine.isPlaying == true, {~stop_func.value;});
				if(((("assist_inst_"++(1)++"_array_part_"++(~assist_part_number)).asSymbol.envirGet != nil) and: (("assist_inst_"++(1)++"_array_part_"++(~assist_part_number)).asSymbol.envirGet != [])), {

					~assist_slider_lenth_func.value;
					AppClock.sched(0.25, {~playbutton.valueAction = 1;});

				});
			});
		);
		("assist_comp_part_"++(~num_assist_parts)++"_button").asSymbol.envirGet.font = Font(~default_font, size: ~firstviewboundsWIDTH/48);

		if(~pieceload != 1, {
			("assist_comp_part_"++(~assist_part_number)++"_button").asSymbol.envirGet.value = 0;
			~assist_part_number = ~num_assist_parts;
			("assist_comp_part_"++(~assist_part_number)++"_button").asSymbol.envirGet.value = 1;
		});
	});
	~new_part_button.font = Font(~default_font, size: ~firstviewboundsWIDTH/48);

	if(~line_multiple1 == nil, {~line_multiple1 = 1});
	if(~line_multiple2 == nil, {~line_multiple2 = 0});

	if(~num_assist_parts == nil, {~num_assist_parts = 0}, {
		~pieceload = 1;
		~num_assist_parts_backup = ~num_assist_parts;
		~line_multiple1_back_up = ~line_multiple1;
		~line_multiple2_back_up = ~line_multiple2;
		~num_assist_parts = 0;
		~line_multiple1 = 1;
		~line_multiple2 = 0;
		~num_assist_parts_backup.do({~new_part_button.valueAction = 0;});
		~num_assist_parts = ~num_assist_parts_backup;
		~line_multiple1 = ~line_multiple1_back_up;
		~line_multiple2 = ~line_multiple2_back_up;
		~pieceload = 0;
	});

	/*~num_assist_parts = 0;
	~line_multiple1 = 0;
	~line_multiple2 = 0;
	~part_limit_counter = 0;*/

	~set_delete_part_button = Button.new(~firstview, Rect(~firstviewboundsWIDTH/72*2 + (~firstviewboundsWIDTH/9), (x=0; ~button_base.value;), ~firstviewboundsWIDTH/6, ~firstviewboundsHEIGHT/21)).states_([["Set Delete Parts",~stringcolor,~backgroundcolor],["Set Delete Parts",~backgroundcolor,Color.green]]).action_({arg button;

		if(button.value == 1, {
			if(~dup_1_part_num != nil, {("assist_comp_part_"++(~dup_1_part_num)++"_button").asSymbol.envirGet.value = 0;});
			if(~dup_2_part_num != nil, {("assist_comp_part_"++(~dup_2_part_num)++"_button").asSymbol.envirGet.value = 0;});
			("assist_comp_part_"++(~assist_part_number)++"_button").asSymbol.envirGet.value = 0;
			~delete_on = 1;
			~delete_array = [];
		}, {
			~delete_on = 0;
			~delete_array.size.do(j=(-1); {
				j=j+1;
				("assist_comp_part_"++(~delete_array.at(j))++"_button").asSymbol.envirGet.value = 0;
			});
		});
	});
	~set_delete_part_button.font = Font(~default_font, size: ~firstviewboundsWIDTH/48);

	~delete_part_button = Button.new(~firstview, Rect(~firstviewboundsWIDTH/72*3 + (~firstviewboundsWIDTH/9) + (~firstviewboundsWIDTH/6), (x=0; ~button_base.value;), ~firstviewboundsWIDTH/8, ~firstviewboundsHEIGHT/21)).states_([["Delete Parts",~stringcolor,~backgroundcolor]]).action_({arg button;

		if(~set_delete_part_button.value == 1, {
			if(~duranduranbot_routine.isPlaying == true, {~stop_func.value;});
			~delete_array.size.do(j=(-1); {
				j=j+1;
				~number_of_instruments.do(s = 0; {
					s=s+1;
					("assist_inst_"++(s)++"_array_part_"++(~delete_array.at(j))++"_backup").asSymbol.envirPut(
						("assist_inst_"++(s)++"_array_part_"++(~delete_array.at(j))).asSymbol.envirGet;
					);
					("assist_inst_"++(s)++"_array_part_"++(~delete_array.at(j))).asSymbol.envirPut(nil);

					("assist_inst_"++(s)++"_comp_interval_array_part_"++(~delete_array.at(j))++"_backup").asSymbol.envirPut(
						("assist_inst_"++(s)++"_comp_interval_array_part_"++(~delete_array.at(j))).asSymbol.envirGet;
					);
					("assist_inst_"++(s)++"_comp_interval_array_part_"++(~delete_array.at(j))).asSymbol.envirPut(nil);

					("assist_inst_"++(s)++"_comp_choice_array_part_"++(~delete_array.at(j))++"_backup").asSymbol.envirPut(
						("assist_inst_"++(s)++"_comp_choice_array_part_"++(~delete_array.at(j))).asSymbol.envirGet;
					);
					("assist_inst_"++(s)++"_comp_choice_array_part_"++(~delete_array.at(j))).asSymbol.envirPut(nil);

				});
			});
			~delete_array.size.do(j=(-1); {
				j=j+1;
				("assist_comp_part_"++(~delete_array.at(j))++"_button").asSymbol.envirGet.value = 0;
			});
			~delete_array_backup = ~delete_array;
			~set_delete_part_button.valueAction = 0;
		});
	});
	~delete_part_button.font = Font(~default_font, size: ~firstviewboundsWIDTH/48);

	~undo_delete_part_button = Button.new(~firstview, Rect(~firstviewboundsWIDTH/72*4 + (~firstviewboundsWIDTH/8)+ (~firstviewboundsWIDTH/9) + (~firstviewboundsWIDTH/6), (x=0; ~button_base.value;), ~firstviewboundsWIDTH/5.3, ~firstviewboundsHEIGHT/21)).states_([["Undo Delete Parts",~stringcolor,~backgroundcolor]]).action_({arg button;

		if(~set_delete_part_button.value == 1, {
			if(~delete_array_backup != [], {
				~delete_array_backup.size.do(j=(-1); {
					j=j+1;
					~number_of_instruments.do(s = 0; {
						s=s+1;
						("assist_inst_"++(s)++"_array_part_"++(~delete_array_backup.at(j))).asSymbol.envirPut(
							("assist_inst_"++(s)++"_array_part_"++(~delete_array_backup.at(j))++"_backup").asSymbol.envirGet;
						);

						("assist_inst_"++(s)++"_comp_interval_array_part_"++(~delete_array_backup.at(j))).asSymbol.envirPut(
							("assist_inst_"++(s)++"_comp_interval_array_part_"++(~delete_array_backup.at(j))++"_backup").asSymbol.envirGet;
						);

						("assist_inst_"++(s)++"_comp_choice_array_part_"++(~delete_array_backup.at(j))).asSymbol.envirPut(
							("assist_inst_"++(s)++"_comp_choice_array_part_"++(~delete_array_backup.at(j))++"_backup").asSymbol.envirGet;
						);

					});
				});
			});
			~set_delete_part_button.valueAction = 0;
		});
	});
	~undo_delete_part_button.font = Font(~default_font, size: ~firstviewboundsWIDTH/48);

	~volbutton_func.value;

	~ml_static = StaticText(~firstview, Rect(~firstviewboundsWIDTH/1.37, (x=1; ~button_base.value;), ~firstviewboundsWIDTH/3.8, ~firstviewboundsHEIGHT/21));
	~ml_static.background_(~backgroundcolor);
	~ml_static.stringColor_(~stringcolor);
	if(~assist_ml_on == 1, {~ml_static.string = "Machine learning is ON";}, {~ml_static.string = "Machine learning is OFF";});
	~ml_static.font = Font(~default_font, size: ~firstviewboundsWIDTH/48);

	~ml_data_static = StaticText(~firstview, Rect(~firstviewboundsWIDTH/1.37, (x=2; ~button_base.value;), ~firstviewboundsWIDTH/3.8, ~firstviewboundsHEIGHT/21));
	~ml_data_static.background_(~backgroundcolor);
	~ml_data_static.stringColor_(~stringcolor);
	if(~assist_ml_data_on == 1, {~ml_data_static.string = "Using ML data is ON";}, {~ml_data_static.string = "Using ML data is OFF";});
	~ml_data_static.font = Font(~default_font, size: ~firstviewboundsWIDTH/48);

	~ai_assist_settings_page_button = Button.new(~firstview, Rect(~firstviewboundsWIDTH/1.37, (x=3; ~button_base.value;), ~firstviewboundsWIDTH/3.8, ~firstviewboundsHEIGHT/21)).states_([["Assist Comp. Settings",~stringcolor,~backgroundcolor]]).action_({arg button;

		~ai_assist_settings_page_func.value;
	});
	~ai_assist_settings_page_button.font = Font(~default_font, size: ~firstviewboundsWIDTH/48);

	~training_settings_page_button = Button.new(~firstview, Rect(~firstviewboundsWIDTH/1.37, (x=4; ~button_base.value;), ~firstviewboundsWIDTH/3.8, ~firstviewboundsHEIGHT/21)).states_([["Player Settings",~stringcolor,~backgroundcolor]]).action_({arg button;
		~player_settings_page_func.value;
	});
	~training_settings_page_button.font = Font(~default_font, size: ~firstviewboundsWIDTH/48);

	~load_piece_to_main_page_button = Button.new(~firstview, Rect(~firstviewboundsWIDTH/1.37, (x=5; ~button_base.value;), ~firstviewboundsWIDTH/3.8, ~firstviewboundsHEIGHT/21)).states_([["Load Piece To Main Page",~stringcolor,~backgroundcolor]]).action_({arg button;

		~number_of_instruments.do(s = 0; {
			s=s+1;
			("instrument_"++(s)++"_full").asSymbol.envirPut([]);
			("instrument_"++(s)++"_comp_interval_array").asSymbol.envirPut([]);
			("instrument_"++(s)++"_comp_choice_array").asSymbol.envirPut([]);
		});
		~full_piece_part_array = [];
		~num_assist_parts.do(j = 0; {
			j=j+1;
			~number_of_instruments.do(s = 0; {
				s=s+1;
				if(("assist_inst_"++(s)++"_array_part_"++(j)).asSymbol.envirGet != nil, {
					("instrument_"++(s)++"_full").asSymbol.envirPut(
						("instrument_"++(s)++"_full").asSymbol.envirGet.add(
							("assist_inst_"++(s)++"_array_part_"++(j)).asSymbol.envirGet;
						);
					);
				});

				if(("assist_inst_"++(s)++"_comp_interval_array_part_"++(j)).asSymbol.envirGet != nil, {
					("instrument_"++(s)++"_comp_interval_array").asSymbol.envirPut(
						("instrument_"++(s)++"_comp_interval_array").asSymbol.envirGet.add(
							("assist_inst_"++(s)++"_comp_interval_array_part_"++(j)).asSymbol.envirGet;
						);
					);
				});

				if(("assist_inst_"++(s)++"_comp_choice_array_part_"++(j)).asSymbol.envirGet != nil, {
					("instrument_"++(s)++"_comp_choice_array").asSymbol.envirPut(
						("instrument_"++(s)++"_comp_choice_array").asSymbol.envirGet.add(
							("assist_inst_"++(s)++"_comp_choice_array_part_"++(j)).asSymbol.envirGet;
						);
					);
				});
			});

			if(("assist_inst_"++(1)++"_array_part_"++(j)).asSymbol.envirGet != nil, {
				~full_piece_part_array = ~full_piece_part_array.add(Array.fill(("assist_inst_"++(1)++"_array_part_"++(j)).asSymbol.envirGet.size, {j}));
			});
		});

		if(("instrument_"++(1)++"_full").asSymbol.envirGet.flatten.at(0).isNumber == true, {

			~full_piece_part_array = ~full_piece_part_array.flatten;

			~number_of_instruments.do(s = 0; {
				s=s+1;
				("instrument_"++(s)++"_full").asSymbol.envirPut(("instrument_"++(s)++"_full").asSymbol.envirGet.flatten);
				("instrument_"++(s)++"_full_midi").asSymbol.envirPut(("instrument_"++(s)++"_full").asSymbol.envirGet.cpsmidi);
				("instrument_"++(s)++"_full_midi").asSymbol.envirPut(("instrument_"++(s)++"_full_midi").asSymbol.envirGet.round);

				("instrument_"++(s)++"_comp_interval_array").asSymbol.envirPut(("instrument_"++(s)++"_comp_interval_array").asSymbol.envirGet.flatten);
				("instrument_"++(s)++"_comp_choice_array").asSymbol.envirPut(("instrument_"++(s)++"_comp_choice_array").asSymbol.envirGet.flatten);
			});
			~create_pseq_arrays_func.value;
			~full_midi_to_ml_func.value;
			~timeslider.controlSpec = ControlSpec(0, ("instrument_"++(1)++"_full").asSymbol.envirGet.size - 1, \lin, 1, 0, "bits");
			"full piece loaded to main page".postln;
		});

		/*if(~load_to_main_play == 1, {~firstviewtext2.valueAction = 0;}); // back button*/
	});
	~load_piece_to_main_page_button.font = Font(~default_font, size: ~firstviewboundsWIDTH/48);

	~train_button_func = {

		var iushaj;

		~train_full_piece_button = Button.new(~firstview, Rect(~firstviewboundsWIDTH/1.37, (iushaj = (x=6); ~button_base.value;), ~firstviewboundsWIDTH/3.8, ~firstviewboundsHEIGHT/21)).states_([["Machine Learn Full Piece",~stringcolor,~backgroundcolor]]).action_({arg button;

			~mlfullpieceprompt = View(~firstview, Rect((~firstviewboundsWIDTH/3), (x = iushaj - 2; ~button_base.value;), ~firstviewboundsWIDTH/3, ~firstviewboundsHEIGHT/3));
			~mlfullpieceprompt.background_(~backgroundcolor);

			~mlfullpieceprompttext = StaticText(~mlfullpieceprompt, Rect(0, 0, ~mlfullpieceprompt.bounds.width, ~mlfullpieceprompt.bounds.height/3));
			~mlfullpieceprompttext.background_(~backgroundcolor);
			~mlfullpieceprompttext.stringColor_(~stringcolor);
			~mlfullpieceprompttext.align = \center;
			~mlfullpieceprompttext.string = "Machine Learn full piece?";
			~mlfullpieceprompttext.font = Font(~default_font, size: ~mlfullpieceprompt.bounds.width/14);
			~mlfullpieceprompttext.front;

			~train_full_piece_yes_no_button_1 = Button.new(~mlfullpieceprompt, Rect(~mlfullpieceprompt.bounds.width/8, ~mlfullpieceprompt.bounds.height/1.6, ~mlfullpieceprompt.bounds.width/4, ~mlfullpieceprompt.bounds.height/5)).states_([["Yes",~stringcolor,~backgroundcolor]]).action_({arg button;

				~load_piece_to_main_page_button.valueAction = 0;
				~full_midi_to_ml_func.value;
				~rate_selection_start = 0;
				~rate_selection_end = ("instrument_"++(1)++"_full_midi_ml").asSymbol.envirGet.size - 1;
				if(~assist_ml_on == 1, {~machine_learning_func.value;});
				("full piece rated:\nx"++(~net_change_mult)++" points\n").postln;
				~mlfullpieceprompt.close;
				Server.default.freeAll;
			});
			~train_full_piece_yes_no_button_1.font = Font(~default_font, size: ~firstviewboundsWIDTH/48);

			~train_full_piece_yes_no_button_2 = Button.new(~mlfullpieceprompt, Rect( ~mlfullpieceprompt.bounds.width/1.6, ~mlfullpieceprompt.bounds.height/1.6, ~mlfullpieceprompt.bounds.width/4, ~mlfullpieceprompt.bounds.height/5)).states_([["No",~stringcolor,~backgroundcolor]]).action_({arg button;

				~mlfullpieceprompt.close;
			});
			~train_full_piece_yes_no_button_2.font = Font(~default_font, size: ~firstviewboundsWIDTH/48);

		});
		~train_full_piece_button.font = Font(~default_font, size: ~firstviewboundsWIDTH/48);

	};

	if(~assist_ml_on == 1, {~train_button_func.value;});

	~ml_gen_func = {

		if(("gen_rout_part_"++(~gen_assist_part_number)).asSymbol.envirGet.isPlaying == true, {
			("gen_rout_part_"++(~gen_assist_part_number)).asSymbol.envirGet.stop;
		});

		("gen_rout_part_"++(~gen_assist_part_number)).asSymbol.envirPut(

			Routine({

				1.do({

					var gen_assist_part_number, last_part_num, next_part_num, e_last, e_next;

					~c_assist_part_number = ~gen_assist_part_number;
					~gen_c_on = 1;
					~last_notes_func.value;
					/*~next_notes_func.value;*/
					~gen_c_on = 0;
					gen_assist_part_number = ~gen_assist_part_number;
					last_part_num = ~last_part_num;
					/*next_part_num = ~next_part_num;*/
					e_last = ~e_last;
					/*e_next = ~e_next;*/

					(60*3).wait;

					if(~assist_ml_on == 1, {

						~number_of_instruments.do(s = 0; {
							s=s+1;
							("instrument_"++(s)++"_full_midi_ml").asSymbol.envirPut(
								if(((e_last != nil) and: (e_last != [])), {(("assist_inst_"++(s)++"_array_part_"++(last_part_num)).asSymbol.envirGet.cpsmidi);})++
								(("assist_inst_"++(s)++"_array_part_"++(gen_assist_part_number)).asSymbol.envirGet.cpsmidi)/*++
								if(((e_next != nil) and: (e_next != [])), {(("assist_inst_"++(s)++"_array_part_"++(next_part_num)).asSymbol.envirGet.cpsmidi);})*/;
							);
							("instrument_"++(s)++"_comp_interval_array_ml").asSymbol.envirPut(
								if(((e_last != nil) and: (e_last != [])), {(("assist_inst_"++(s)++"_comp_interval_array_part_"++(last_part_num)).asSymbol.envirGet;);})++
								(("assist_inst_"++(s)++"_comp_interval_array_part_"++(gen_assist_part_number)).asSymbol.envirGet;)/*++
								if(((e_next != nil) and: (e_next != [])), {(("assist_inst_"++(s)++"_comp_interval_array_part_"++(next_part_num)).asSymbol.envirGet;);})*/;
							);
							("instrument_"++(s)++"_comp_choice_array_ml").asSymbol.envirPut(
								if(((e_last != nil) and: (e_last != [])), {(("assist_inst_"++(s)++"_comp_choice_array_part_"++(last_part_num)).asSymbol.envirGet;);})++
								(("assist_inst_"++(s)++"_comp_choice_array_part_"++(gen_assist_part_number)).asSymbol.envirGet;)/*++
								if(((e_next != nil) and: (e_next != [])), {(("assist_inst_"++(s)++"_comp_choice_array_part_"++(next_part_num)).asSymbol.envirGet;);})*/;
							);
						});

						~rate_selection_start = 0;
						~rate_selection_end = ("instrument_"++(1)++"_full_midi_ml").asSymbol.envirGet.size - 1;
						~dup_ml_on = 1;
						~machine_learning_func.value;
						~dup_ml_on = 0;
					});
				});
			});
		);

		("gen_rout_part_"++(~gen_assist_part_number)).asSymbol.envirGet.play;
	};

	~generate_new_button = Button.new(~firstview, Rect(~firstviewboundsWIDTH/72, (x=2; ~button_base.value;), ~firstviewboundsWIDTH/6.5, ~firstviewboundsHEIGHT/21)).states_([["Generate New",~stringcolor,~backgroundcolor]]).action_({arg button;

		if(~duranduranbot_routine.isPlaying == true, {~stop_func.value;});
		~number_of_instruments.do(s = 0; {
			s=s+1;
			("assist_inst_"++(s)++"_array_part_"++(~assist_part_number)++"_gen_backup1").asSymbol.envirPut(
				("assist_inst_"++(s)++"_array_part_"++(~assist_part_number)).asSymbol.envirGet;
			);

			("assist_inst_"++(s)++"_comp_interval_array_part_"++(~assist_part_number)++"_gen_backup1").asSymbol.envirPut(
				("assist_inst_"++(s)++"_comp_interval_array_part_"++(~assist_part_number)).asSymbol.envirGet;
			);

			("assist_inst_"++(s)++"_comp_choice_array_part_"++(~assist_part_number)++"_gen_backup1").asSymbol.envirPut(
				("assist_inst_"++(s)++"_comp_choice_array_part_"++(~assist_part_number)).asSymbol.envirGet;
			);
		});

		~assisted_comp_part_func.value;

		AppClock.sched(0.25, {

			~gen_new = 1;
			~gen_assist_part_number = ~assist_part_number;
			~assist_slider_lenth_func.value;
			~playbutton.valueAction = 1;

		});
	});
	~generate_new_button.font = Font(~default_font, size: ~firstviewboundsWIDTH/48);

	~undo_generate_new_button = Button.new(~firstview, Rect(~firstviewboundsWIDTH/72*2 + (~firstviewboundsWIDTH/6.5), (x=2; ~button_base.value;), ~firstviewboundsWIDTH/4.7, ~firstviewboundsHEIGHT/21)).states_([["Undo Generate New",~stringcolor,~backgroundcolor]]).action_({arg button;

		if(~duranduranbot_routine.isPlaying == true, {~stop_func.value;});
		~number_of_instruments.do(s = 0; {
			s=s+1;
			("assist_inst_"++(s)++"_array_part_"++(~assist_part_number)++"_gen_backup2").asSymbol.envirPut(
				("assist_inst_"++(s)++"_array_part_"++(~assist_part_number)).asSymbol.envirGet;
			);
			("assist_inst_"++(s)++"_array_part_"++(~assist_part_number)).asSymbol.envirPut(
				("assist_inst_"++(s)++"_array_part_"++(~assist_part_number)++"_gen_backup1").asSymbol.envirGet;
			);
			("assist_inst_"++(s)++"_array_part_"++(~assist_part_number)++"_gen_backup1").asSymbol.envirPut(
				("assist_inst_"++(s)++"_array_part_"++(~assist_part_number)++"_gen_backup2").asSymbol.envirGet;
			);

			("assist_inst_"++(s)++"_comp_interval_array_part_"++(~assist_part_number)++"_gen_backup2").asSymbol.envirPut(
				("assist_inst_"++(s)++"_comp_interval_array_part_"++(~assist_part_number)).asSymbol.envirGet;
			);
			("assist_inst_"++(s)++"_comp_interval_array_part_"++(~assist_part_number)).asSymbol.envirPut(
				("assist_inst_"++(s)++"_comp_interval_array_part_"++(~assist_part_number)++"_gen_backup1").asSymbol.envirGet;
			);
			("assist_inst_"++(s)++"_comp_interval_array_part_"++(~assist_part_number)++"_gen_backup1").asSymbol.envirPut(
				("assist_inst_"++(s)++"_comp_interval_array_part_"++(~assist_part_number)++"_gen_backup2").asSymbol.envirGet;
			);

			("assist_inst_"++(s)++"_comp_choice_array_part_"++(~assist_part_number)++"_gen_backup2").asSymbol.envirPut(
				("assist_inst_"++(s)++"_comp_choice_array_part_"++(~assist_part_number)).asSymbol.envirGet;
			);
			("assist_inst_"++(s)++"_comp_choice_array_part_"++(~assist_part_number)).asSymbol.envirPut(
				("assist_inst_"++(s)++"_comp_choice_array_part_"++(~assist_part_number)++"_gen_backup1").asSymbol.envirGet;
			);
			("assist_inst_"++(s)++"_comp_choice_array_part_"++(~assist_part_number)++"_gen_backup1").asSymbol.envirPut(
				("assist_inst_"++(s)++"_comp_choice_array_part_"++(~assist_part_number)++"_gen_backup2").asSymbol.envirGet;
			);

		});

		AppClock.sched(0.25, {

			~gen_new = 1;
			~gen_assist_part_number = ~assist_part_number;
			~assist_slider_lenth_func.value;
			~playbutton.valueAction = 1;

		});
	});
	~undo_generate_new_button.font = Font(~default_font, size: ~firstviewboundsWIDTH/48);

	// ~save_generate_new_button = Button.new(~firstview, Rect(~firstviewboundsWIDTH/72*3 + (~firstviewboundsWIDTH/7) + (~firstviewboundsWIDTH/5), ~firstviewboundsHEIGHT/2.97, ~firstviewboundsWIDTH/5, ~firstviewboundsHEIGHT/21)).states_([["Save Generate New",~stringcolor,~backgroundcolor]]).action_({arg button;
	//
	// });
	// ~save_generate_new_button.font = Font(~default_font, size: ~firstviewboundsWIDTH/48);

	~partavgdurstatictext_1 = StaticText(~firstview, Rect(~firstviewboundsWIDTH/72, (x=1; ~button_base.value;), ~firstviewboundsWIDTH/4.2, ~firstviewboundsHEIGHT/21));
	~partavgdurstatictext_1.background_(~backgroundcolor);
	~partavgdurstatictext_1.stringColor_(~stringcolor);
	~partavgdurstatictext_1.string = "Avg. part length (secs.) -";
	~partavgdurstatictext_1.font = Font(~default_font, size: ~firstviewboundsWIDTH/48);

	~partavgdurtextfield_1 = TextField(~firstview, Rect(~firstviewboundsWIDTH/72+(~firstviewboundsWIDTH/4.2)+(~firstviewboundsWIDTH/280), (x=1; ~button_base.value;), ~firstviewboundsWIDTH/10, ~firstviewboundsHEIGHT/21));
	~partavgdurtextfield_1.string = ~avg_part_length.asString;
	~partavgdurtextfield_1.background_(~backgroundcolor);
	~partavgdurtextfield_1.stringColor_(~stringcolor);
	~partavgdurtextfield_1.font = Font(~default_font, size: ~firstviewboundsWIDTH/48);
	~partavgdurtextfield_1.action = {
		~sdfk = ~partavgdurtextfield_1.value.asFloat;
		/*	~pos_1 = looptextfield_1.value.asInteger;*/
		if(~sdfk.value.isNumber == true, {
			~avg_part_length = ~sdfk.value.asFloat;
			~assist_avg_part_length = ~sdfk.value.asFloat;
			~partavgdurtextfield_1.string = ~avg_part_length.asString;
			~avg_part_length.postln;
			g = File((~ddbot_root)++"/program_files/other_program_files/assist_avg_part_length.scd", "w");
			g.write(~avg_part_length.asString;);
			g.close;
		}, {~partavgdurtextfield_1.string = "number".asString;});
	};

	~partavgdurstatictext_2 = StaticText(~firstview, Rect((~firstviewboundsWIDTH/72*2)+(~firstviewboundsWIDTH/4.2)+(~firstviewboundsWIDTH/280)+(~firstviewboundsWIDTH/10), (x=1; ~button_base.value;), ~firstviewboundsWIDTH/7.6, ~firstviewboundsHEIGHT/21));
	~partavgdurstatictext_2.background_(~backgroundcolor);
	~partavgdurstatictext_2.stringColor_(~stringcolor);
	~partavgdurstatictext_2.string = "Range (+/-) -";
	~partavgdurstatictext_2.font = Font(~default_font, size: ~firstviewboundsWIDTH/48);

	~partavgdurtextfield_2 = TextField(~firstview, Rect((~firstviewboundsWIDTH/72*2)+(~firstviewboundsWIDTH/4.2)+(~firstviewboundsWIDTH/7.6)+(~firstviewboundsWIDTH/14)+(~firstviewboundsWIDTH/25), (x=1; ~button_base.value;), ~firstviewboundsWIDTH/10, ~firstviewboundsHEIGHT/21));
	~partavgdurtextfield_2.string = ~avg_part_range.asString;
	~partavgdurtextfield_2.background_(~backgroundcolor);
	~partavgdurtextfield_2.stringColor_(~stringcolor);
	~partavgdurtextfield_2.font = Font(~default_font, size: ~firstviewboundsWIDTH/48);
	~partavgdurtextfield_2.action = {
		~sdfk = ~partavgdurtextfield_2.value.asFloat;
		/*	~pos_1 = looptextfield_1.value.asInteger;*/
		if(~sdfk.value.isNumber == true, {
			~avg_part_range = ~sdfk.value.asFloat;
			~assist_avg_part_range = ~sdfk.value.asFloat;
			~partavgdurtextfield_2.string = ~avg_part_range.asString;
			~avg_part_range.postln;
			g = File((~ddbot_root)++"/program_files/other_program_files/assist_avg_part_range.scd", "w");
			g.write(~avg_part_range.asString;);
			g.close;
		}, {~partavgdurtextfield_2.string = "number".asString;});
	};

	~set_duplicate_button = Button.new(~firstview, Rect(~firstviewboundsWIDTH/72, (x=3; ~button_base.value;), ~firstviewboundsWIDTH/6.6, ~firstviewboundsHEIGHT/21)).states_([["Set Duplicate",~stringcolor,~backgroundcolor],["Set Duplicate",~backgroundcolor,Color.green]]).action_({arg button;

		if(~dup_1_part_num != nil, {("assist_comp_part_"++(~dup_1_part_num)++"_button").asSymbol.envirGet.value = 0;});
		if(~dup_2_part_num != nil, {("assist_comp_part_"++(~dup_2_part_num)++"_button").asSymbol.envirGet.value = 0;});
		if(button.value == 1, {~duplicate_on = 1}, {
			~duplicate_on = 0;
			~dup_1_part_num = nil;
		});
	});
	~set_duplicate_button.font = Font(~default_font, size: ~firstviewboundsWIDTH/48);

	if(~duplicate_on == 1, {~set_duplicate_button.value = 1;});

	~duplicate_button = Button.new(~firstview, Rect(~firstviewboundsWIDTH/72*2 + (~firstviewboundsWIDTH/6.6), (x=3; ~button_base.value;), ~firstviewboundsWIDTH/6.6, ~firstviewboundsHEIGHT/21)).states_([["Duplicate",~stringcolor,~backgroundcolor]]).action_({arg button;

		if(((~duplicate_on == 1) and: (~dup_1_part_num != nil) and: (~dup_2_part_num != nil)), {
			~number_of_instruments.do(s = 0; {
				s=s+1;
				("assist_inst_"++(s)++"_array_part_"++(~dup_2_part_num)++"_dup_save_1").asSymbol.envirPut(
					("assist_inst_"++(s)++"_array_part_"++(~dup_2_part_num)).asSymbol.envirGet;
				);
				("assist_inst_"++(s)++"_array_part_"++(~dup_2_part_num)).asSymbol.envirPut(
					("assist_inst_"++(s)++"_array_part_"++(~dup_1_part_num)).asSymbol.envirGet;
				);

				("assist_inst_"++(s)++"_comp_interval_array_part_"++(~dup_2_part_num)++"_dup_save_1").asSymbol.envirPut(
					("assist_inst_"++(s)++"_comp_interval_array_part_"++(~dup_2_part_num)).asSymbol.envirGet;
				);
				("assist_inst_"++(s)++"_comp_interval_array_part_"++(~dup_2_part_num)).asSymbol.envirPut(
					("assist_inst_"++(s)++"_comp_interval_array_part_"++(~dup_1_part_num)).asSymbol.envirGet;
				);

				("assist_inst_"++(s)++"_comp_choice_array_part_"++(~dup_2_part_num)++"_dup_save_1").asSymbol.envirPut(
					("assist_inst_"++(s)++"_comp_choice_array_part_"++(~dup_2_part_num)).asSymbol.envirGet;
				);
				("assist_inst_"++(s)++"_comp_choice_array_part_"++(~dup_2_part_num)).asSymbol.envirPut(
					("assist_inst_"++(s)++"_comp_choice_array_part_"++(~dup_1_part_num)).asSymbol.envirGet;
				);

				if(~assist_ml_on == 1, {

					// machine learning training on duplicate

					("instrument_"++(s)++"_full_midi_ml").asSymbol.envirPut(
						("assist_inst_"++(s)++"_array_part_"++(~dup_1_part_num)).asSymbol.envirGet.cpsmidi;
					);
					("instrument_"++(s)++"_comp_interval_array_ml").asSymbol.envirPut(
						("assist_inst_"++(s)++"_comp_interval_array_part_"++(~dup_1_part_num)).asSymbol.envirGet;
					);
					("instrument_"++(s)++"_comp_choice_array_ml").asSymbol.envirPut(
						("assist_inst_"++(s)++"_comp_choice_array_part_"++(~dup_1_part_num)).asSymbol.envirGet;
					);
				});

			});

			if(~assist_ml_on == 1, {
				~rate_selection_start = 0;
				~rate_selection_end = ("instrument_"++(1)++"_full_midi_ml").asSymbol.envirGet.size - 1;
				~dup_ml_on = 1;
				~machine_learning_func.value;
				~dup_ml_on = 0;
			});

			("assist_comp_part_"++(~dup_1_part_num)++"_button").asSymbol.envirGet.value = 0;
			("assist_comp_part_"++(~dup_2_part_num)++"_button").asSymbol.envirGet.value = 0;
			~dup_1_part_num = nil;
			~duplicate_on = 0;
			~set_duplicate_button.value = 0;
		}, {"set duplicate not on\n - and/or no 2 parts where selected".postln;});
	});
	~duplicate_button.font = Font(~default_font, size: ~firstviewboundsWIDTH/48);

	~undo_duplicate_button = Button.new(~firstview, Rect(~firstviewboundsWIDTH/72*3 + (~firstviewboundsWIDTH/6.6*2), (x=3; ~button_base.value;), ~firstviewboundsWIDTH/6.1, ~firstviewboundsHEIGHT/21)).states_([["Undo Duplicate",~stringcolor,~backgroundcolor]]).action_({arg button;

		if(~set_duplicate_button.value == 1, {
			~number_of_instruments.do(s = 0; {
				s=s+1;
				//uses reverse forumla found in "~reversenewrandomseedsbutton" on main page
				("assist_inst_"++(s)++"_array_part_"++(~dup_2_part_num)++"_dup_save_2").asSymbol.envirPut(
					("assist_inst_"++(s)++"_array_part_"++(~dup_2_part_num)).asSymbol.envirGet;
				);
				("assist_inst_"++(s)++"_array_part_"++(~dup_2_part_num)).asSymbol.envirPut(
					("assist_inst_"++(s)++"_array_part_"++(~dup_2_part_num)++"_dup_save_1").asSymbol.envirGet;
				);
				("assist_inst_"++(s)++"_array_part_"++(~dup_2_part_num)++"_dup_save_1").asSymbol.envirPut(
					("assist_inst_"++(s)++"_array_part_"++(~dup_2_part_num)++"_dup_save_2").asSymbol.envirGet;
				);

				("assist_inst_"++(s)++"_comp_interval_array_part_"++(~dup_2_part_num)++"_dup_save_2").asSymbol.envirPut(
					("assist_inst_"++(s)++"_comp_interval_array_part_"++(~dup_2_part_num)).asSymbol.envirGet;
				);
				("assist_inst_"++(s)++"_comp_interval_array_part_"++(~dup_2_part_num)).asSymbol.envirPut(
					("assist_inst_"++(s)++"_comp_interval_array_part_"++(~dup_2_part_num)++"_dup_save_1").asSymbol.envirGet;
				);
				("assist_inst_"++(s)++"_comp_interval_array_part_"++(~dup_2_part_num)++"_dup_save_1").asSymbol.envirPut(
					("assist_inst_"++(s)++"_comp_interval_array_part_"++(~dup_2_part_num)++"_dup_save_2").asSymbol.envirGet;
				);

				("assist_inst_"++(s)++"_comp_choice_array_part_"++(~dup_2_part_num)++"_dup_save_2").asSymbol.envirPut(
					("assist_inst_"++(s)++"_comp_choice_array_part_"++(~dup_2_part_num)).asSymbol.envirGet;
				);
				("assist_inst_"++(s)++"_comp_choice_array_part_"++(~dup_2_part_num)).asSymbol.envirPut(
					("assist_inst_"++(s)++"_comp_choice_array_part_"++(~dup_2_part_num)++"_dup_save_1").asSymbol.envirGet;
				);
				("assist_inst_"++(s)++"_comp_choice_array_part_"++(~dup_2_part_num)++"_dup_save_1").asSymbol.envirPut(
					("assist_inst_"++(s)++"_comp_choice_array_part_"++(~dup_2_part_num)++"_dup_save_2").asSymbol.envirGet;
				);

				"undo duplicate performed".postln;

			});
			~set_duplicate_button.valueAction = 0;
		});
	});
	~undo_duplicate_button.font = Font(~default_font, size: ~firstviewboundsWIDTH/48);

	if(~batch_dup_1 == nil, {~batch_dup_1 = "";});
	if(~batch_dup_2 == nil, {~batch_dup_2 = "";});
	if(~batch_dup_3 == nil, {~batch_dup_3 = "";});
	if(~batch_dup_4 == nil, {~batch_dup_4 = "";});

	~batchduptext = StaticText(~firstview, Rect(~firstviewboundsWIDTH/72, (x=4; ~button_base.value;), ~firstviewboundsWIDTH/6.2, ~firstviewboundsHEIGHT/21));
	~batchduptext.background=~backgroundcolor;
	~batchduptext.stringColor=~stringcolor;
	~batchduptext.string = "Batch Duplicate -";
	~batchduptext.font = Font(~default_font, size: ~firstviewboundsWIDTH/48);

	~batchduptext1 = StaticText(~firstview, Rect(~firstviewboundsWIDTH/72, (x=5; ~button_base.value;), ~firstviewboundsWIDTH/6.5, ~firstviewboundsHEIGHT/21));
	~batchduptext1.background=~backgroundcolor;
	~batchduptext1.stringColor=~stringcolor;
	~batchduptext1.string = "To dup. 1st/last";
	~batchduptext1.font = Font(~default_font, size: ~firstviewboundsWIDTH/48);

	~batchduptextfield_1 = TextField(~firstview, Rect(~firstviewboundsWIDTH/72+(~firstviewboundsWIDTH/6.2)+(~firstviewboundsWIDTH/280), (x=5; ~button_base.value;), ~firstviewboundsWIDTH/17, ~firstviewboundsHEIGHT/21));
	~batchduptextfield_1.string = ~batch_dup_1.asString;
	~batchduptextfield_1.background_(~backgroundcolor);
	~batchduptextfield_1.stringColor_(~stringcolor);
	~batchduptextfield_1.font = Font(~default_font, size: ~firstviewboundsWIDTH/48);
	~batchduptextfield_1.action = {
		~sdfk = ~batchduptextfield_1.value.asInteger;
		if(~sdfk.value.isInteger == true, {
			~batch_dup_1 = ~sdfk.value.asInteger;
			~batchduptextfield_1.string = ~batch_dup_1.asString;
			~batch_dup_1.postln;
		}, {~batchduptextfield_1.string = "int.".asString;});
	};

	~batchduptextfield_2 = TextField(~firstview, Rect(~firstviewboundsWIDTH/72+(~firstviewboundsWIDTH/6.2)+(~firstviewboundsWIDTH/17)+(~firstviewboundsWIDTH/280*2), (x=5; ~button_base.value;), ~firstviewboundsWIDTH/17, ~firstviewboundsHEIGHT/21));
	~batchduptextfield_2.string = ~batch_dup_2.asString;
	~batchduptextfield_2.background_(~backgroundcolor);
	~batchduptextfield_2.stringColor_(~stringcolor);
	~batchduptextfield_2.font = Font(~default_font, size: ~firstviewboundsWIDTH/48);
	~batchduptextfield_2.action = {
		~sdfk = ~batchduptextfield_2.value.asInteger;
		if(~sdfk.value.isInteger == true, {
			~batch_dup_2 = ~sdfk.value.asInteger;
			~batchduptextfield_2.string = ~batch_dup_2.asString;
			~batch_dup_2.postln;
		}, {~batchduptextfield_2.string = "int.".asString;});
	};

	~batchduptext2 = StaticText(~firstview, Rect(~firstviewboundsWIDTH/72, (x=6; ~button_base.value;), ~firstviewboundsWIDTH/6, ~firstviewboundsHEIGHT/21));
	~batchduptext2.background=~backgroundcolor;
	~batchduptext2.stringColor=~stringcolor;
	~batchduptext2.string = "Dup. into 1st/last";
	~batchduptext2.font = Font(~default_font, size: ~firstviewboundsWIDTH/48);

	~batchduptextfield_3 = TextField(~firstview, Rect(~firstviewboundsWIDTH/72+(~firstviewboundsWIDTH/6)+(~firstviewboundsWIDTH/280*1), (x=6; ~button_base.value;), ~firstviewboundsWIDTH/17, ~firstviewboundsHEIGHT/21));
	~batchduptextfield_3.string = ~batch_dup_3.asString;
	~batchduptextfield_3.background_(~backgroundcolor);
	~batchduptextfield_3.stringColor_(~stringcolor);
	~batchduptextfield_3.font = Font(~default_font, size: ~firstviewboundsWIDTH/48);
	~batchduptextfield_3.action = {
		~sdfk = ~batchduptextfield_3.value.asInteger;
		if(~sdfk.value.isInteger == true, {
			~batch_dup_3 = ~sdfk.value.asInteger;
			~batchduptextfield_3.string = ~batch_dup_3.asString;
			~batch_dup_3.postln;
		}, {~batchduptextfield_3.string = "int.".asString;});
	};

	~batchduptextfield_4 = TextField(~firstview, Rect(~firstviewboundsWIDTH/72+(~firstviewboundsWIDTH/6)+(~firstviewboundsWIDTH/17)+(~firstviewboundsWIDTH/280*2), (x=6; ~button_base.value;), ~firstviewboundsWIDTH/17, ~firstviewboundsHEIGHT/21));
	~batchduptextfield_4.string = ~batch_dup_4.asString;
	~batchduptextfield_4.background_(~backgroundcolor);
	~batchduptextfield_4.stringColor_(~stringcolor);
	~batchduptextfield_4.font = Font(~default_font, size: ~firstviewboundsWIDTH/48);
	~batchduptextfield_4.action = {
		~sdfk = ~batchduptextfield_4.value.asInteger;
		if(~sdfk.value.isInteger == true, {
			~batch_dup_4 = ~sdfk.value.asInteger;
			~batchduptextfield_4.string = ~batch_dup_4.asString;
			~batch_dup_4.postln;
		}, {~batchduptextfield_4.string = "int.".asString;});
	};

	~batch_duplicate_button = Button.new(~firstview, Rect(~firstviewboundsWIDTH/72+(~firstviewboundsWIDTH/6)+(~firstviewboundsWIDTH/17*2)+(~firstviewboundsWIDTH/280*4)+(~firstviewboundsWIDTH/17/2), (x=6; ~button_base.value;), ~firstviewboundsWIDTH/11, ~firstviewboundsHEIGHT/21)).states_([["Set Dup.",~stringcolor,~backgroundcolor]]).action_({arg button;

		var num_dups1, num_dups2, dup_num;

		if(~assist_ml_on == 1, {
			~number_of_instruments.do(s = 0; {
				s=s+1;
				("instrument_"++(s)++"_full_midi_ml").asSymbol.envirPut([]);
				("instrument_"++(s)++"_comp_interval_array_ml").asSymbol.envirPut([]);
				("instrument_"++(s)++"_comp_choice_array_ml").asSymbol.envirPut([]);
			});
		});

		num_dups1 = ~batch_dup_2 - ~batch_dup_1 + 1;
		num_dups2 = ~batch_dup_4 - ~batch_dup_3 + 1;
		if((
			(num_dups1.isNegative == false) and:
			(num_dups2.isNegative == false) and:
			(num_dups1 == num_dups2) and:
			(num_dups1 != 0) and:
			(num_dups2 != 0) and:
			(num_dups1 != nil) and:
			(num_dups2 != nil) and:
			(~batch_dup_1 != nil) and:
			(~batch_dup_2 != nil) and:
			(~batch_dup_3 != nil) and:
			(~batch_dup_4 != nil) and:
			(~batch_dup_1 != 0) and:
			(~batch_dup_2 != 0) and:
			(~batch_dup_3 != 0) and:
			(~batch_dup_4 != 0)
		), {
			num_dups2.do(dup_num=(-1); {
				dup_num=dup_num+1;
				if(dup_num <= num_dups2, {
					~number_of_instruments.do(s = 0; {
						s=s+1;

						("assist_inst_"++(s)++"_array_part_"++(~batch_dup_3+dup_num)++"batch_dup_save_1").asSymbol.envirPut(
							("assist_inst_"++(s)++"_array_part_"++(~batch_dup_3+dup_num)).asSymbol.envirGet;
						);
						("assist_inst_"++(s)++"_array_part_"++(~batch_dup_3+dup_num)).asSymbol.envirPut(
							("assist_inst_"++(s)++"_array_part_"++(~batch_dup_1+dup_num)).asSymbol.envirGet;
						);

						("assist_inst_"++(s)++"_comp_interval_array_part_"++(~batch_dup_3+dup_num)++"batch_dup_save_1").asSymbol.envirPut(
							("assist_inst_"++(s)++"_comp_interval_array_part_"++(~batch_dup_3+dup_num)).asSymbol.envirGet;
						);
						("assist_inst_"++(s)++"_comp_interval_array_part_"++(~batch_dup_3+dup_num)).asSymbol.envirPut(
							("assist_inst_"++(s)++"_comp_interval_array_part_"++(~batch_dup_1+dup_num)).asSymbol.envirGet;
						);

						("assist_inst_"++(s)++"_comp_choice_array_part_"++(~batch_dup_3+dup_num)++"batch_dup_save_1").asSymbol.envirPut(
							("assist_inst_"++(s)++"_comp_choice_array_part_"++(~batch_dup_3+dup_num)).asSymbol.envirGet;
						);
						("assist_inst_"++(s)++"_comp_choice_array_part_"++(~batch_dup_3+dup_num)).asSymbol.envirPut(
							("assist_inst_"++(s)++"_comp_choice_array_part_"++(~batch_dup_1+dup_num)).asSymbol.envirGet;
						);

						if(~assist_ml_on == 1, {

							// machine learning training on duplicate

							("instrument_"++(s)++"_full_midi_ml").asSymbol.envirPut(
								("instrument_"++(s)++"_full_midi_ml").asSymbol.envirGet.add(
									("assist_inst_"++(s)++"_array_part_"++(~batch_dup_1+dup_num)).asSymbol.envirGet.cpsmidi;
								)
							);
							("instrument_"++(s)++"_comp_interval_array_ml").asSymbol.envirPut(
								("instrument_"++(s)++"_comp_interval_array_ml").asSymbol.envirGet.add(
									("assist_inst_"++(s)++"_comp_interval_array_part_"++(~batch_dup_1+dup_num)).asSymbol.envirGet;
								)
							);
							("instrument_"++(s)++"_comp_choice_array_ml").asSymbol.envirPut(
								("instrument_"++(s)++"_comp_choice_array_ml").asSymbol.envirGet.add(
									("assist_inst_"++(s)++"_comp_choice_array_part_"++(~batch_dup_1+dup_num)).asSymbol.envirGet;
								)
							);
						});

					});

				});
			});

			if(~assist_ml_on == 1, {

				~number_of_instruments.do(s = 0; {
					s=s+1;
					("instrument_"++(s)++"_full_midi_ml").asSymbol.envirPut(
						("instrument_"++(s)++"_full_midi_ml").asSymbol.envirGet.flatten;
					);
					("instrument_"++(s)++"_comp_interval_array_ml").asSymbol.envirPut(
						("instrument_"++(s)++"_comp_interval_array_ml").asSymbol.envirGet.flatten;
					);
					("instrument_"++(s)++"_comp_choice_array_ml").asSymbol.envirPut(
						("instrument_"++(s)++"_comp_choice_array_ml").asSymbol.envirGet.flatten;
					);
				});

				~rate_selection_start = 0;
				~rate_selection_end = ("instrument_"++(1)++"_full_midi_ml").asSymbol.envirGet.size - 1;
				~dup_ml_on = 1;
				~machine_learning_func.value;
				~dup_ml_on = 0;
			});

			~num_dups_save = num_dups2;
			~batch_dup_3_save = ~batch_dup_3;
			"batch duplicate performed".postln;
		}, {var statictext;
			"error occured. try again.".postln;
			/*statictext = StaticText(~firstview, Rect(~firstviewboundsWIDTH/72+(~firstviewboundsWIDTH/6.2*2)+(~firstviewboundsWIDTH/17*3)+(~firstviewboundsWIDTH/6)+(~firstviewboundsWIDTH/280*6), ~firstviewboundsHEIGHT/1.88, ~firstviewboundsWIDTH/17, ~firstviewboundsHEIGHT/21));
			statictext.string = ~batch_dup_4.asString;
			statictext.background_(~backgroundcolor);
			statictext.stringColor_(~stringcolor);
			statictext.font = Font(~default_font, size: ~firstviewboundsWIDTH/48);*/
		});
	});
	~batch_duplicate_button.font = Font(~default_font, size: ~firstviewboundsWIDTH/48);

	~undo_batch_duplicate_button = Button.new(~firstview, Rect(~firstviewboundsWIDTH/72+(~firstviewboundsWIDTH/6)+(~firstviewboundsWIDTH/17*2)+(~firstviewboundsWIDTH/280*9)+(~firstviewboundsWIDTH/17*2), (x=6; ~button_base.value;), ~firstviewboundsWIDTH/15, ~firstviewboundsHEIGHT/21)).states_([["Undo",~stringcolor,~backgroundcolor]]).action_({arg button;

		if(~num_dups_save != nil, {
			~num_dups_save.do(j=(-1); {
				j=j+1;
				if(j <= ~num_dups_save, {
					~number_of_instruments.do(s = 0; {
						s=s+1;
						//uses reverse forumla found in "~reversenewrandomseedsbutton" on main page
						("assist_inst_"++(s)++"_array_part_"++(~batch_dup_3_save+j)++"batch_dup_save_2").asSymbol.envirPut(
							("assist_inst_"++(s)++"_array_part_"++(~batch_dup_3_save+j)).asSymbol.envirGet;
						);
						("assist_inst_"++(s)++"_array_part_"++(~batch_dup_3_save+j)).asSymbol.envirPut(
							("assist_inst_"++(s)++"_array_part_"++(~batch_dup_3_save+j)++"batch_dup_save_1").asSymbol.envirGet;
						);
						("assist_inst_"++(s)++"_array_part_"++(~batch_dup_3_save+j)++"batch_dup_save_1").asSymbol.envirPut(
							("assist_inst_"++(s)++"_array_part_"++(~batch_dup_3_save+j)++"batch_dup_save_2").asSymbol.envirGet;
						);

						("assist_inst_"++(s)++"_comp_interval_array_part_"++(~batch_dup_3_save+j)++"batch_dup_save_2").asSymbol.envirPut(
							("assist_inst_"++(s)++"_comp_interval_array_part_"++(~batch_dup_3_save+j)).asSymbol.envirGet;
						);
						("assist_inst_"++(s)++"_comp_interval_array_part_"++(~batch_dup_3_save+j)).asSymbol.envirPut(
							("assist_inst_"++(s)++"_comp_interval_array_part_"++(~batch_dup_3_save+j)++"batch_dup_save_1").asSymbol.envirGet;
						);
						("assist_inst_"++(s)++"_comp_interval_array_part_"++(~batch_dup_3_save+j)++"batch_dup_save_1").asSymbol.envirPut(
							("assist_inst_"++(s)++"_comp_interval_array_part_"++(~batch_dup_3_save+j)++"batch_dup_save_2").asSymbol.envirGet;
						);

						("assist_inst_"++(s)++"_comp_choice_array_part_"++(~batch_dup_3_save+j)++"batch_dup_save_2").asSymbol.envirPut(
							("assist_inst_"++(s)++"_comp_choice_array_part_"++(~batch_dup_3_save+j)).asSymbol.envirGet;
						);
						("assist_inst_"++(s)++"_comp_choice_array_part_"++(~batch_dup_3_save+j)).asSymbol.envirPut(
							("assist_inst_"++(s)++"_comp_choice_array_part_"++(~batch_dup_3_save+j)++"batch_dup_save_1").asSymbol.envirGet;
						);
						("assist_inst_"++(s)++"_comp_choice_array_part_"++(~batch_dup_3_save+j)++"batch_dup_save_1").asSymbol.envirPut(
							("assist_inst_"++(s)++"_comp_choice_array_part_"++(~batch_dup_3_save+j)++"batch_dup_save_2").asSymbol.envirGet;
						);

					});
				});
			});

			"undo batch duplicate performed".postln;
		});
	});
	~undo_batch_duplicate_button.font = Font(~default_font, size: ~firstviewboundsWIDTH/48);

	~mergetext = StaticText(~firstview, Rect(~firstviewboundsWIDTH/72, (x=8; ~button_base.value;), ~firstviewboundsWIDTH/13, ~firstviewboundsHEIGHT/21));
	~mergetext.background=~backgroundcolor;
	~mergetext.stringColor=~stringcolor;
	~mergetext.string = "Merge - ";
	~mergetext.font = Font(~default_font, size: ~firstviewboundsWIDTH/48);

	~mergetext1 = StaticText(~firstview, Rect(~firstviewboundsWIDTH/72*2+(~firstviewboundsWIDTH/13), (x=8; ~button_base.value;), ~firstviewboundsWIDTH/7, ~firstviewboundsHEIGHT/21));
	~mergetext1.background=~backgroundcolor;
	~mergetext1.stringColor=~stringcolor;
	~mergetext1.string = "Beginning part";
	~mergetext1.font = Font(~default_font, size: ~firstviewboundsWIDTH/48);

	if(~merge_beg_part == nil, {~merge_beg_part = "";});
	if(~merge_end_part == nil, {~merge_end_part = "";});
	if(~merge_into_part == nil, {~merge_into_part = "";});

	~mergetextfield_1 = TextField(~firstview, Rect(~firstviewboundsWIDTH/72*3+(~firstviewboundsWIDTH/13)+(~firstviewboundsWIDTH/7), (x=8; ~button_base.value;), ~firstviewboundsWIDTH/17, ~firstviewboundsHEIGHT/21));
	~mergetextfield_1.string = ~merge_beg_part.asString;
	~mergetextfield_1.background_(~backgroundcolor);
	~mergetextfield_1.stringColor_(~stringcolor);
	~mergetextfield_1.font = Font(~default_font, size: ~firstviewboundsWIDTH/48);
	~mergetextfield_1.action = {
		~sdfk = ~mergetextfield_1.value.asFloat;
		/*	~pos_1 = looptextfield_1.value.asInteger;*/
		if(~sdfk.value.isNumber == true, {
			~merge_beg_part = ~sdfk.value.asInteger;
			~mergetextfield_1.string = ~merge_beg_part.asString;
			~merge_beg_part.postln;
		}, {~mergetextfield_1.string = "number".asString;});
	};

	~mergetext2 = StaticText(~firstview, Rect(~firstviewboundsWIDTH/72*4+(~firstviewboundsWIDTH/13)+(~firstviewboundsWIDTH/7)+(~firstviewboundsWIDTH/17), (x=8; ~button_base.value;), ~firstviewboundsWIDTH/8.5, ~firstviewboundsHEIGHT/21));
	~mergetext2.background=~backgroundcolor;
	~mergetext2.stringColor=~stringcolor;
	~mergetext2.string = "Ending part";
	~mergetext2.font = Font(~default_font, size: ~firstviewboundsWIDTH/48);

	~mergetextfield_2 = TextField(~firstview, Rect(~firstviewboundsWIDTH/72*5+(~firstviewboundsWIDTH/13)+(~firstviewboundsWIDTH/7)+(~firstviewboundsWIDTH/17)+(~firstviewboundsWIDTH/8.5), (x=8; ~button_base.value;), ~firstviewboundsWIDTH/17, ~firstviewboundsHEIGHT/21));
	~mergetextfield_2.string = ~merge_end_part.asString;
	~mergetextfield_2.background_(~backgroundcolor);
	~mergetextfield_2.stringColor_(~stringcolor);
	~mergetextfield_2.font = Font(~default_font, size: ~firstviewboundsWIDTH/48);
	~mergetextfield_2.action = {
		~sdfk = ~mergetextfield_2.value.asFloat;
		/*	~pos_1 = looptextfield_1.value.asInteger;*/
		if(~sdfk.value.isNumber == true, {
			~merge_end_part = ~sdfk.value.asInteger;
			~mergetextfield_2.string = ~merge_end_part.asString;
			~merge_end_part.postln;
		}, {~mergetextfield_2.string = "number".asString;});
	};

	~mergetext3 = StaticText(~firstview, Rect(~firstviewboundsWIDTH/72, (x=9; ~button_base.value;), ~firstviewboundsWIDTH/6.5, ~firstviewboundsHEIGHT/21));
	~mergetext3.background=~backgroundcolor;
	~mergetext3.stringColor=~stringcolor;
	~mergetext3.string = "Merge into part";
	~mergetext3.font = Font(~default_font, size: ~firstviewboundsWIDTH/48);

	~mergetextfield_3 = TextField(~firstview, Rect(~firstviewboundsWIDTH/72*2+(~firstviewboundsWIDTH/6.5), (x=9; ~button_base.value;), ~firstviewboundsWIDTH/17, ~firstviewboundsHEIGHT/21));
	~mergetextfield_3.string = ~merge_into_part.asString;
	~mergetextfield_3.background_(~backgroundcolor);
	~mergetextfield_3.stringColor_(~stringcolor);
	~mergetextfield_3.font = Font(~default_font, size: ~firstviewboundsWIDTH/48);
	~mergetextfield_3.action = {
		~sdfk = ~mergetextfield_3.value.asFloat;
		/*	~pos_1 = looptextfield_1.value.asInteger;*/
		if(~sdfk.value.isNumber == true, {
			~merge_into_part = ~sdfk.value.asInteger;
			~mergetextfield_3.string = ~merge_into_part.asString;
			~merge_into_part.postln;
		}, {~mergetextfield_3.string = "number".asString;});
	};

	~set_merge_button = Button.new(~firstview, Rect(~firstviewboundsWIDTH/72*3+(~firstviewboundsWIDTH/6.5)+(~firstviewboundsWIDTH/17), (x=9; ~button_base.value;), ~firstviewboundsWIDTH/9, ~firstviewboundsHEIGHT/21)).states_([["Set Merge",~stringcolor,~backgroundcolor]]).action_({arg button;

		if(~duranduranbot_routine.isPlaying == true, {~stop_func.value;});

		~number_of_instruments.do(s = 0; {
			s=s+1;

			("assist_inst_"++(s)++"_array_part_"++(~merge_into_part)++"_merge_save_1").asSymbol.envirPut(
				("assist_inst_"++(s)++"_array_part_"++(~merge_into_part)).asSymbol.envirGet;
			);
			("assist_inst_"++(s)++"_comp_interval_array_part_"++(~merge_into_part)++"_merge_save_1").asSymbol.envirPut(
				("assist_inst_"++(s)++"_comp_interval_array_part_"++(~merge_into_part)).asSymbol.envirGet;
			);
			("assist_inst_"++(s)++"_comp_choice_array_part_"++(~merge_into_part)++"_merge_save_1").asSymbol.envirPut(
				("assist_inst_"++(s)++"_comp_choice_array_part_"++(~merge_into_part)).asSymbol.envirGet;
			);


			("assist_inst_"++(s)++"_array_part_"++(~merge_into_part)).asSymbol.envirPut(
				("assist_inst_"++(s)++"_array_part_"++(~merge_beg_part)).asSymbol.envirGet++
				("assist_inst_"++(s)++"_array_part_"++(~merge_end_part)).asSymbol.envirGet;
			);
			("assist_inst_"++(s)++"_comp_interval_array_part_"++(~merge_into_part)).asSymbol.envirPut(
				("assist_inst_"++(s)++"_comp_interval_array_part_"++(~merge_beg_part)).asSymbol.envirGet++
				("assist_inst_"++(s)++"_comp_interval_array_part_"++(~merge_end_part)).asSymbol.envirGet;
			);
			("assist_inst_"++(s)++"_comp_choice_array_part_"++(~merge_into_part)).asSymbol.envirPut(
				("assist_inst_"++(s)++"_comp_choice_array_part_"++(~merge_beg_part)).asSymbol.envirGet++
				("assist_inst_"++(s)++"_comp_choice_array_part_"++(~merge_end_part)).asSymbol.envirGet;
			);

		});

		if(~assist_ml_on == 1, {

			~number_of_instruments.do(s = 0; {
				s=s+1;
				("instrument_"++(s)++"_full_midi_ml").asSymbol.envirPut(
					("assist_inst_"++(s)++"_array_part_"++(~merge_into_part)).asSymbol.envirGet.cpsmidi;
				);
				("instrument_"++(s)++"_comp_interval_array_ml").asSymbol.envirPut(
					("assist_inst_"++(s)++"_comp_interval_array_part_"++(~merge_into_part)).asSymbol.envirGet;
				);
				("instrument_"++(s)++"_comp_choice_array_ml").asSymbol.envirPut(
					("assist_inst_"++(s)++"_comp_choice_array_part_"++(~merge_into_part)).asSymbol.envirGet;
				);
			});

			~rate_selection_start = 0;
			~rate_selection_end = ("instrument_"++(1)++"_full_midi_ml").asSymbol.envirGet.size - 1;
			~dup_ml_on = 1;
			~machine_learning_func.value;
			~dup_ml_on = 0;
		});

		("merge into part"++(~merge_into_part)++" performed").postln;
	});
	~set_merge_button.font = Font(~default_font, size: ~firstviewboundsWIDTH/48);

	~undo_merge_button = Button.new(~firstview, Rect(~firstviewboundsWIDTH/72*4+(~firstviewboundsWIDTH/6.5)+(~firstviewboundsWIDTH/17)+( ~firstviewboundsWIDTH/9), (x=9; ~button_base.value;), ~firstviewboundsWIDTH/12, ~firstviewboundsHEIGHT/21)).states_([["Undo",~stringcolor,~backgroundcolor]]).action_({arg button;

		if(~duranduranbot_routine.isPlaying == true, {~stop_func.value;});

		~number_of_instruments.do(s = 0; {
			s=s+1;

			("assist_inst_"++(s)++"_array_part_"++(~merge_into_part)++"_merge_save_2").asSymbol.envirPut(
				("assist_inst_"++(s)++"_array_part_"++(~merge_into_part)).asSymbol.envirGet;
			);
			("assist_inst_"++(s)++"_array_part_"++(~merge_into_part)).asSymbol.envirPut(
				("assist_inst_"++(s)++"_array_part_"++(~merge_into_part)++"_merge_save_1").asSymbol.envirGet;
			);
			("assist_inst_"++(s)++"_array_part_"++(~merge_into_part)++"_merge_save_1").asSymbol.envirPut(
				("assist_inst_"++(s)++"_array_part_"++(~merge_into_part)++"_merge_save_2").asSymbol.envirGet;
			);

			("assist_inst_"++(s)++"_comp_interval_array_part_"++(~merge_into_part)++"_merge_save_2").asSymbol.envirPut(
				("assist_inst_"++(s)++"_comp_interval_array_part_"++(~merge_into_part)).asSymbol.envirGet;
			);
			("assist_inst_"++(s)++"_comp_interval_array_part_"++(~merge_into_part)).asSymbol.envirPut(
				("assist_inst_"++(s)++"_comp_interval_array_part_"++(~merge_into_part)++"_merge_save_1").asSymbol.envirGet;
			);
			("assist_inst_"++(s)++"_comp_interval_array_part_"++(~merge_into_part)++"_merge_save_1").asSymbol.envirPut(
				("assist_inst_"++(s)++"_comp_interval_array_part_"++(~merge_into_part)++"_merge_save_2").asSymbol.envirGet;
			);

			("assist_inst_"++(s)++"_comp_choice_array_part_"++(~merge_into_part)++"_merge_save_2").asSymbol.envirPut(
				("assist_inst_"++(s)++"_comp_choice_array_part_"++(~merge_into_part)).asSymbol.envirGet;
			);
			("assist_inst_"++(s)++"_comp_choice_array_part_"++(~merge_into_part)).asSymbol.envirPut(
				("assist_inst_"++(s)++"_comp_choice_array_part_"++(~merge_into_part)++"_merge_save_1").asSymbol.envirGet;
			);
			("assist_inst_"++(s)++"_comp_choice_array_part_"++(~merge_into_part)++"_merge_save_1").asSymbol.envirPut(
				("assist_inst_"++(s)++"_comp_choice_array_part_"++(~merge_into_part)++"_merge_save_2").asSymbol.envirGet;
			);

		});

		("undo merge into part"++(~merge_into_part)++" performed").postln;
	});
	~undo_merge_button.font = Font(~default_font, size: ~firstviewboundsWIDTH/48);

	~set_up_trim_button = Button.new(~firstview, Rect(~firstviewboundsWIDTH/72, (x=11; ~button_base.value;), ~firstviewboundsWIDTH/9, ~firstviewboundsHEIGHT/21)).states_([["Trim OFF",~stringcolor,~backgroundcolor],["Trim ON",~backgroundcolor,Color.green]]).action_({arg button;

		~test_trim = 0;
		~test_trim2 = 0;

		if(button.value == 1, {

			~number_of_instruments.do(s = 0; {
				s=s+1;

				("assist_inst_"++(s)++"_array_part_"++(~assist_part_number)++"_trim_original").asSymbol.envirPut(
					("assist_inst_"++(s)++"_array_part_"++(~assist_part_number)).asSymbol.envirGet;
				);

				("assist_inst_"++(s)++"_comp_interval_array_part_"++(~assist_part_number)++"_trim_original").asSymbol.envirPut(
					("assist_inst_"++(s)++"_comp_interval_array_part_"++(~assist_part_number)).asSymbol.envirGet;
				);

				("assist_inst_"++(s)++"_comp_choice_array_part_"++(~assist_part_number)++"_trim_original").asSymbol.envirPut(
					("assist_inst_"++(s)++"_comp_choice_array_part_"++(~assist_part_number)).asSymbol.envirGet;
				);
			});

			~set_trim_on = 1;
			if(~trimviewtext3 == nil, {
				~trimviewtext3 = StaticText(~firstview, Rect(~firstviewboundsWIDTH/72, (x=10; ~button_base.value;), ~firstviewboundsWIDTH/1.75, ~firstviewboundsHEIGHT/21));
				~trimviewtext3.background=~backgroundcolor;
				~trimviewtext3.stringColor_(~stringcolor);
				~trimviewtext3.string = "";
				~trimviewtext3.font = Font(~default_font, size: ~firstviewboundsWIDTH/48);
			});
		}, {~set_trim_on = 0; ~trimviewtext3.close; ~trimviewtext3 = nil;});
	});
	~set_up_trim_button.font = Font(~default_font, size: ~firstviewboundsWIDTH/48);

	if(~set_trim_on == 1, {~set_up_trim_button.value = 1;});

	~trimviewtext1 = StaticText(~firstview, Rect(~firstviewboundsWIDTH/72*2+(~firstviewboundsWIDTH/9), (x=11; ~button_base.value;), ~firstviewboundsWIDTH/2, ~firstviewboundsHEIGHT/21));
	~trimviewtext1.background=~backgroundcolor;
	~trimviewtext1.stringColor_(~stringcolor);
	~trimviewtext1.string = "- recommend keeping off if not in use";
	~trimviewtext1.font = Font(~default_font, size: ~firstviewboundsWIDTH/48);

	if(~cutslider_1_position == nil, {~cutslider_1_position = 0;});
	if(~cutslider_2_position == nil, {~cutslider_2_position = 0;});

	~cut_case_func = {
		case
		{~cutslider_1_position > ~cutslider_2_position}{~cut_from = ~cutslider_2_position + 1; ~cut_to = ~cutslider_1_position + 1;}
		{~cutslider_1_position < ~cutslider_2_position}{~cut_from = ~cutslider_1_position + 1; ~cut_to = ~cutslider_2_position + 1;}
		{~cutslider_1_position == ~cutslider_2_position}{~cut_from = ~cutslider_1_position + 1; ~cut_to = ~cutslider_1_position + 1;};
	};

	~cutslider1 = EZSlider(~firstview,  bounds: (Rect(~firstviewboundsWIDTH/72, (x=12; ~button_base.value;), ~firstviewboundsWIDTH/1.5, ~firstviewboundsHEIGHT/21)), controlSpec: ControlSpec(0, ("instrument_"++(1)++"_full").asSymbol.envirGet.size - 1, \lin, 1, 0, "bits"), action: {
		|timeselect|
		var selection1;
		if(~set_trim_on == 1, {
			~cutslider_1_position = timeselect.value;
			~cut_case_func.value;
			~trimviewtext3.string = "Keep from bit "++(~cut_from)++" to "++(~cut_to)++", and discard the rest.";
		});

	});
	~cutslider1.setColors(~stringcolor,~stringcolor);

	~cutslider2 = EZSlider(~firstview,  bounds: (Rect(~firstviewboundsWIDTH/72, (x=13; ~button_base.value;), ~firstviewboundsWIDTH/1.5, ~firstviewboundsHEIGHT/21)), controlSpec: ControlSpec(0, ("instrument_"++(1)++"_full").asSymbol.envirGet.size - 1, \lin, 1, 0, "bits"), action: {
		|timeselect|
		var selection1;
		if(~set_trim_on == 1, {
			~cutslider_2_position = timeselect.value;
			~cut_case_func.value;
			~trimviewtext3.string = "Keep from bit "++(~cut_from)++" to "++(~cut_to)++", and discard the rest.";
		});
	});
	~cutslider2.setColors(~stringcolor,~stringcolor);

	~trim_play_button = Button.new(~firstview, Rect(~firstviewboundsWIDTH/72, (x=14; ~button_base.value;), ~firstviewboundsWIDTH/9, ~firstviewboundsHEIGHT/21)).states_([["Play part",~stringcolor,~backgroundcolor]]).action_({arg button;

		if(~set_trim_on == 1, {
			if(("assist_inst_"++(1)++"_array_part_"++(~assist_part_number)).asSymbol.envirGet.size != nil, {
				~assist_slider_lenth_func.value;
				~playbutton.valueAction = 1;
			});
		});
	});
	~trim_play_button.font = Font(~default_font, size: ~firstviewboundsWIDTH/48);

	~test_trim_button = Button.new(~firstview, Rect(~firstviewboundsWIDTH/72*2+(~firstviewboundsWIDTH/9), (x=14; ~button_base.value;), ~firstviewboundsWIDTH/9, ~firstviewboundsHEIGHT/21)).states_([["Test Trim",~stringcolor,~backgroundcolor]]).action_({arg button;

		if(~set_trim_on == 1, {
			if(("assist_inst_"++(1)++"_array_part_"++(~assist_part_number)).asSymbol.envirGet.size != nil, {

				~test_trim = 1;
				~cut_from_old = ~cut_from;
				~assist_slider_lenth_func.value;
				~playbutton.valueAction = 1;
			});
		});
	});
	~test_trim_button.font = Font(~default_font, size: ~firstviewboundsWIDTH/48);

	~trim_stop_button = Button.new(~firstview, Rect(~firstviewboundsWIDTH/72+(~firstviewboundsWIDTH/9*5), (x=14; ~button_base.value;), ~firstviewboundsWIDTH/9, ~firstviewboundsHEIGHT/21)).states_([["Stop",~stringcolor,~backgroundcolor]]).action_({arg button;

		if(~set_trim_on == 1, {
			if(~duranduranbot_routine.isPlaying == true, {~stop_func.value;});
		});
	});
	~trim_stop_button.font = Font(~default_font, size: ~firstviewboundsWIDTH/48);

	~trim_button = Button.new(~firstview, Rect(~firstviewboundsWIDTH/72*3+(~firstviewboundsWIDTH/9*2), (x=14; ~button_base.value;), ~firstviewboundsWIDTH/15, ~firstviewboundsHEIGHT/21)).states_([["Trim",~stringcolor,~backgroundcolor]]).action_({arg button;

		if(~set_trim_on == 1, {
			~save_assist_project_button.valueAction = 0;
			if(~trimviewtext2 != nil, {~trimviewtext2.close; ~trimviewtext2 = nil;});
			~trimviewtext2 = StaticText(~firstview, Rect(~firstviewboundsWIDTH/72*3+(~firstviewboundsWIDTH/9*2) +(~firstviewboundsWIDTH/14), (x=14; ~button_base.value;), ~firstviewboundsWIDTH/6, ~firstviewboundsHEIGHT/21));
			~trimviewtext2.background=~backgroundcolor;
			~trimviewtext2.stringColor_(~stringcolor);
			~trimviewtext2.string = "- old proj. saved";
			~trimviewtext2.font = Font(~default_font, size: ~firstviewboundsWIDTH/48);
			AppClock.sched(3, {~trimviewtext2.close; ~trimviewtext2 = nil;});

			if(~duranduranbot_routine.isPlaying == true, {~stop_func.value;});

			~number_of_instruments.do(s = 0; {
				s=s+1;

				("assist_inst_"++(s)++"_array_part_"++(~assist_part_number)++"_trim_save_1").asSymbol.envirPut(
					("assist_inst_"++(s)++"_array_part_"++(~assist_part_number)).asSymbol.envirGet;
				);

				("assist_inst_"++(s)++"_comp_interval_array_part_"++(~assist_part_number)++"_trim_save_1").asSymbol.envirPut(
					("assist_inst_"++(s)++"_comp_interval_array_part_"++(~assist_part_number)).asSymbol.envirGet;
				);

				("assist_inst_"++(s)++"_comp_choice_array_part_"++(~assist_part_number)++"_trim_save_1").asSymbol.envirPut(
					("assist_inst_"++(s)++"_comp_choice_array_part_"++(~assist_part_number)).asSymbol.envirGet;
				);


				(~cut_to - ~cut_from + 1).do(t = ~cut_from - 2; ~new_array = []; {
					~new_array = ~new_array.add(
						("assist_inst_"++(s)++"_array_part_"++(~assist_part_number)).asSymbol.envirGet.at(t=t+1);
					);
				});
				("assist_inst_"++(s)++"_array_part_"++(~assist_part_number)).asSymbol.envirPut(~new_array);

				(~cut_to - ~cut_from + 1).do(t = ~cut_from - 2; ~new_array = []; {
					~new_array = ~new_array.add(
						("assist_inst_"++(s)++"_comp_interval_array_part_"++(~assist_part_number)).asSymbol.envirGet.at(t=t+1);
					);
				});
				("assist_inst_"++(s)++"_comp_interval_array_part_"++(~assist_part_number)).asSymbol.envirPut(~new_array);

				(~cut_to - ~cut_from + 1).do(t = ~cut_from - 2; ~new_array = []; {
					~new_array = ~new_array.add(
						("assist_inst_"++(s)++"_comp_choice_array_part_"++(~assist_part_number)).asSymbol.envirGet.at(t=t+1);
					);
				});
				("assist_inst_"++(s)++"_comp_choice_array_part_"++(~assist_part_number)).asSymbol.envirPut(~new_array);

			});

			if(~assist_ml_on == 1, {

				~number_of_instruments.do(s = 0; {
					s=s+1;
					("instrument_"++(s)++"_full_midi_ml").asSymbol.envirPut(
						("assist_inst_"++(s)++"_array_part_"++(~assist_part_number)).asSymbol.envirGet.cpsmidi;
					);
					("instrument_"++(s)++"_comp_interval_array_ml").asSymbol.envirPut(
						("assist_inst_"++(s)++"_comp_interval_array_part_"++(~assist_part_number)).asSymbol.envirGet;
					);
					("instrument_"++(s)++"_comp_choice_array_ml").asSymbol.envirPut(
						("assist_inst_"++(s)++"_comp_choice_array_part_"++(~assist_part_number)).asSymbol.envirGet;
					);
				});

				~rate_selection_start = 0;
				~rate_selection_end = ("instrument_"++(1)++"_full_midi_ml").asSymbol.envirGet.size - 1;
				~dup_ml_on = 1;
				~machine_learning_func.value;
				~dup_ml_on = 0;
			});

			"trim performed".postln;
		});

	});
	~trim_button.font = Font(~default_font, size: ~firstviewboundsWIDTH/48);

	~undo_trim_button = Button.new(~firstview, Rect(~firstviewboundsWIDTH/72, (x=15; ~button_base.value;), ~firstviewboundsWIDTH/15, ~firstviewboundsHEIGHT/21)).states_([["Undo",~stringcolor,~backgroundcolor]]).action_({arg button;

		if(~set_trim_on == 1, {

			if(~duranduranbot_routine.isPlaying == true, {~stop_func.value;});

			~number_of_instruments.do(s = 0; {
				s=s+1;

				("assist_inst_"++(s)++"_array_part_"++(~assist_part_number)++"_trim_save_2").asSymbol.envirPut(
					("assist_inst_"++(s)++"_array_part_"++(~assist_part_number)).asSymbol.envirGet;
				);
				("assist_inst_"++(s)++"_array_part_"++(~assist_part_number)).asSymbol.envirPut(
					("assist_inst_"++(s)++"_array_part_"++(~assist_part_number)++"_trim_save_1").asSymbol.envirGet;
				);
				("assist_inst_"++(s)++"_array_part_"++(~assist_part_number)++"_trim_save_1").asSymbol.envirPut(
					("assist_inst_"++(s)++"_array_part_"++(~assist_part_number)++"_trim_save_2").asSymbol.envirGet;
				);

				("assist_inst_"++(s)++"_comp_interval_array_part_"++(~assist_part_number)++"_trim_save_2").asSymbol.envirPut(
					("assist_inst_"++(s)++"_comp_interval_array_part_"++(~assist_part_number)).asSymbol.envirGet;
				);
				("assist_inst_"++(s)++"_comp_interval_array_part_"++(~assist_part_number)).asSymbol.envirPut(
					("assist_inst_"++(s)++"_comp_interval_array_part_"++(~assist_part_number)++"_trim_save_1").asSymbol.envirGet;
				);
				("assist_inst_"++(s)++"_comp_interval_array_part_"++(~assist_part_number)++"_trim_save_1").asSymbol.envirPut(
					("assist_inst_"++(s)++"_comp_interval_array_part_"++(~assist_part_number)++"_trim_save_2").asSymbol.envirGet;
				);

				("assist_inst_"++(s)++"_comp_choice_array_part_"++(~assist_part_number)++"_trim_save_2").asSymbol.envirPut(
					("assist_inst_"++(s)++"_comp_choice_array_part_"++(~assist_part_number)).asSymbol.envirGet;
				);
				("assist_inst_"++(s)++"_comp_choice_array_part_"++(~assist_part_number)).asSymbol.envirPut(
					("assist_inst_"++(s)++"_comp_choice_array_part_"++(~assist_part_number)++"_trim_save_1").asSymbol.envirGet;
				);
				("assist_inst_"++(s)++"_comp_choice_array_part_"++(~assist_part_number)++"_trim_save_1").asSymbol.envirPut(
					("assist_inst_"++(s)++"_comp_choice_array_part_"++(~assist_part_number)++"_trim_save_2").asSymbol.envirGet;
				);

			});

			"undo trim performed".postln;
		});
	});
	~undo_trim_button.font = Font(~default_font, size: ~firstviewboundsWIDTH/48);

	~set_back_original_trim_button = Button.new(~firstview, Rect(~firstviewboundsWIDTH/72*2+(~firstviewboundsWIDTH/15), (x=15; ~button_base.value;), ~firstviewboundsWIDTH/5, ~firstviewboundsHEIGHT/21)).states_([["Set back to original",~stringcolor,~backgroundcolor]]).action_({arg button;

		if(~set_trim_on == 1, {

			if(~duranduranbot_routine.isPlaying == true, {~stop_func.value;});

			~number_of_instruments.do(s = 0; {
				s=s+1;

				("assist_inst_"++(s)++"_array_part_"++(~assist_part_number)++"_trim_save_1").asSymbol.envirPut(
					("assist_inst_"++(s)++"_array_part_"++(~assist_part_number)).asSymbol.envirGet;
				);
				("assist_inst_"++(s)++"_array_part_"++(~assist_part_number)).asSymbol.envirPut(
					("assist_inst_"++(s)++"_array_part_"++(~assist_part_number)++"_trim_original").asSymbol.envirGet;
				);

				("assist_inst_"++(s)++"_comp_interval_array_part_"++(~assist_part_number)++"_trim_save_1").asSymbol.envirPut(
					("assist_inst_"++(s)++"_comp_interval_array_part_"++(~assist_part_number)).asSymbol.envirGet;
				);
				("assist_inst_"++(s)++"_comp_interval_array_part_"++(~assist_part_number)).asSymbol.envirPut(
					("assist_inst_"++(s)++"_comp_interval_array_part_"++(~assist_part_number)++"_trim_original").asSymbol.envirGet;
				);

				("assist_inst_"++(s)++"_comp_choice_array_part_"++(~assist_part_number)++"_trim_save_1").asSymbol.envirPut(
					("assist_inst_"++(s)++"_comp_choice_array_part_"++(~assist_part_number)).asSymbol.envirGet;
				);
				("assist_inst_"++(s)++"_comp_choice_array_part_"++(~assist_part_number)).asSymbol.envirPut(
					("assist_inst_"++(s)++"_comp_choice_array_part_"++(~assist_part_number)++"_trim_original").asSymbol.envirGet;
				);
			});

			"original before trim performed".postln;
		});
	});
	~set_back_original_trim_button.font = Font(~default_font, size: ~firstviewboundsWIDTH/48);

	/*fillertext = StaticText(~firstview, Rect(0, ~firstviewboundsHEIGHT/0.999999, ~firstviewboundsWIDTH/2.2, ~firstviewboundsHEIGHT/21));
	fillertext.background=~backgroundcolor;
	fillertext.stringColor=~stringcolor;
	fillertext.string = "";*/

	/*fillertext2 = StaticText(~firstview2, Rect(0, ~firstviewboundsHEIGHT/0.999999, ~firstviewboundsWIDTH/2.2, ~firstviewboundsHEIGHT/21));
	fillertext2.background=~backgroundcolor;
	fillertext2.stringColor=~stringcolor;
	fillertext2.string = "";*/

};