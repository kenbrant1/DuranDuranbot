~main_page_func = {

	var arrow, newrandomseedsbutton, reversenewrandomseedsbutton, currentrandomseedstrext, playnil, see_saved_seeds_file_button;

	~ai_assist_compose = 0;

	~full_midi_to_ml_func.value;

	~avg_part_length = ~ai_avg_part_length;
	~avg_part_range = ~ai_avg_part_range;

	if(~firstview != nil, {~firstview.close;});
	if(~firstview2 != nil, {~firstview2.close;});

	~firstview = ScrollView(w, Rect(0, 0, w.bounds.width, w.bounds.height));
	/*view.clearOnRefresh = false;*/
	~firstview.background = ~backgroundcolor;
	~firstviewboundsWIDTH = ~firstview.bounds.width;
	~firstviewboundsHEIGHT = ~firstview.bounds.height;

	~ddbot_assist_compose_button = Button.new(~firstview, Rect(~firstviewboundsWIDTH/72, ~firstviewboundsHEIGHT/50, ~firstviewboundsWIDTH/4.2, ~firstviewboundsHEIGHT/21)).states_([["DDbot Assist Compose",Color(0.0, 1.0, 1.0),~backgroundcolor]]).action_({arg button;

		~firstview.close;
		~assist_compose_page_func.value;

	});
	~ddbot_assist_compose_button.font = Font(~default_font, size: ~firstviewboundsWIDTH/48);

	~firstviewtext1 = StaticText(~firstview, Rect(~firstviewboundsWIDTH/2.2857142857143, ~firstviewboundsHEIGHT/43.7, ~firstviewboundsWIDTH/8, ~firstviewboundsHEIGHT/21));
	~firstviewtext1.background=~backgroundcolor;
	~firstviewtext1.stringColor_(~stringcolor);
	~firstviewtext1.string = "Main Page";
	~firstviewtext1.align = \topLeft;
	~firstviewtext1.font = Font(~default_font, size: ~firstviewboundsWIDTH/48);

	~firstviewtext3 = Button.new(~firstview, Rect(~firstviewboundsWIDTH/1.4173228346457, ~firstviewboundsHEIGHT/43.7, ~firstviewboundsWIDTH/3.7, ~firstviewboundsHEIGHT/21)).states_([["Instructions/Help",~stringcolor,~backgroundcolor]]).action_({arg button;
		~instructions_page_func.value;

	});
	~firstviewtext3.font = Font(~default_font, size: ~firstviewboundsWIDTH/48);

	~fullscreen_func = {

		w.close;
		w = Window("DuranDuranbot",
			Rect(0, 0, Window.availableBounds.width, Window.availableBounds.height),

			/*Rect(0, 0, Window.screenBounds.width, Window.screenBounds.height),*/
			resizable: true, border:true);

		w.front; w.fullScreen;
		if(~ai_assist_compose != 1, {~main_page_func.value;}, {~assist_compose_page_func.value;});
		~fullscreenbutton.value = 1;
		~fullscreenon = 1;
		g = File((~ddbot_root)++"/program_files/other_program_files/fullscreenon.scd", "w");
		g.write("1");
		g.close;

		~hotkeys_func.value;

	};

	~end_fullscreen_func = {

		w.close;
		w = Window("DuranDuranbot",
			Rect(Window.availableBounds.width*(1/4.4), Window.availableBounds.height*(1/4.5), Window.availableBounds.width*((4.5-2)/4.5), Window.availableBounds.height*((4.5-2)/4.5)),

			/*Rect(0, 0, Window.screenBounds.width, Window.screenBounds.height),*/
			resizable: true, border:true);

		w.front; /*w.fullScreen;*/
		if(~ai_assist_compose != 1, {~main_page_func.value;}, {~assist_compose_page_func.value;});
		~fullscreenbutton.value = 0;
		~fullscreenon = 0;
		g = File((~ddbot_root)++"/program_files/other_program_files/fullscreenon.scd", "w");
		g.write("0");
		g.close;

		~hotkeys_func.value;

	};

	~fullscreenbutton = Button.new(~firstview, Rect(~firstviewboundsWIDTH/1.4173228346457, ~firstviewboundsHEIGHT/43.7+(~firstviewboundsHEIGHT/21), ~firstviewboundsWIDTH/3.7, ~firstviewboundsHEIGHT/21)).states_([["Enter Fullscreen",~stringcolor,~backgroundcolor],["Exit Fullscreen",~stringcolor,~backgroundcolor]]).action_({arg button;

		if(button.value == 1, {~fullscreen_func.value;}, {~end_fullscreen_func.value;});

	});
	~fullscreenbutton.font = Font(~default_font, size: ~firstviewboundsWIDTH/48);

	if(~fullscreenon == 1, {~fullscreenbutton.value = 1;});

	~button_base = {(~firstviewboundsHEIGHT/6.24286+(((~firstviewboundsHEIGHT/3)-(~firstviewboundsHEIGHT/3.85))*x))};

	~generatepiecebutton = Button.new(~firstview, Rect(~firstviewboundsWIDTH/72, (x=0; ~button_base.value;), ~firstviewboundsWIDTH/6, ~firstviewboundsHEIGHT/21)).states_([["Generate Piece",~stringcolor,~backgroundcolor]]).action_({arg button;
		var piecegenerated;
		~piecegenerated = StaticText(~firstview, Rect(~firstviewboundsWIDTH/72, ~firstviewboundsHEIGHT/9.5, ~firstviewboundsWIDTH/6.2608695652174, ~firstviewboundsWIDTH/36));
		~piecegenerated.background=~backgroundcolor;
		~piecegenerated.stringColor_(~stringcolor);
		"Please Wait".postln;
		~piecegenerated.font = Font(~default_font, size: ~firstviewboundsWIDTH/55);
		~piecegenerated.string = "Please Wait";/*
		if(~piece_created == "true", {~playbutton.close; ~arrow.close;});*/
		~piece_generator_func.value;

	});
	~generatepiecebutton.font = Font(~default_font, size: ~firstviewboundsWIDTH/48);

	/*if(playnil == nil, {nil}, {*/
	/*~play_visible_func = {
	~play_visible_func_verify*/

	~playbutton_func = {
		~playbutton = Button.new(~firstview, Rect(~firstviewboundsWIDTH/(if(~ai_assist_compose == 0, {4.5}, {4.65})), if(~ai_assist_compose == 0, {(x=0; ~button_base.value;)}, {~firstviewboundsHEIGHT/7.5}), ~firstviewboundsWIDTH/8, ~firstviewboundsHEIGHT/21)).states_([["Play Piece",~stringcolor,~backgroundcolor],["Pause Piece",~stringcolor,~backgroundcolor]]).action_({arg button;
			var pause_piece;
			if(button.value == 1, {
				if(~pause_piece != 1, {
					~seedscheck = 1;
					~stopbuttonfunc.value;
					case
					{(~ai_assist_compose == 0) or: (~load_assist == 1)}{~run_piece_func.value;}
					{~ai_assist_compose == 1}{~assisted_ai_run_piece_func.value;}
				}, {/*~timer_routine.resume;*/
					Server.default.makeBundle(Server.default.latency, {~number_of_instruments.do(j = 0; {("duranduranbotsynth"++(j=j+1)).asSymbol.envirGet.set(\amp, ~amp);});});
					~duranduranbot_routine.resume;
					"Piece Resumed".postln;
					if(~will_record == 1, {Server.default.record;});

				});
				~pause_piece = 0;
			}, {
				/*~timer_routine.pause;*/
				Server.default.makeBundle(Server.default.latency, {~number_of_instruments.do(j = 0; {("duranduranbotsynth"++(j=j+1)).asSymbol.envirGet.set(\amp, 0);});});
				~duranduranbot_routine.pause;
				/*Server.default.makeBundle(Server.default.latency, {~number_of_instruments.do(x = 0; {("duranduranbotsynth"++(x=x+1)).asSymbol.envirGet.set(\freq, 0);});});
				*/if(~ai_assist_compose != 1, {"Piece Paused".postln;});
				/*if(~will_record == 1, {Server.default.pauseRecording;});*/
				~pause_piece = 1;
				~playbutton.string = "Resume";
			});

		});
		~playbutton.font = Font(~default_font, size: ~firstviewboundsWIDTH/48);
		/*};*/
		~arrow = StaticText(~firstview, Rect(~firstviewboundsWIDTH/(if(~ai_assist_compose == 0, {5.25}, {5.5})), if(~ai_assist_compose == 0, {(x=0; ~button_base.value;)}, {~firstviewboundsHEIGHT/7.5}), ~firstviewboundsWIDTH/40, ~firstviewboundsHEIGHT/21));
		~arrow.background=~backgroundcolor;
		~arrow.stringColor_(~stringcolor);
		~arrow.string = "---";
		~arrow.font = Font(~default_font, size: ~firstviewboundsWIDTH/48);
	};

	~playbutton_func.value;
	if(~pause_piece == 1, {~playbutton.string = "Resume"; ~stopbuttonfunc.value;});

	if(~piece_created == nil, {~playbutton.close; ~arrow.close;});

	~stop_func = {
		if(~green != nil, {("assist_comp_part_"++(~green)++"_button").asSymbol.envirGet.value = 0; ~green = nil;});
		/*if(~set_trim_on == 1, {~test_trim_stop_func.value});*/
		~duranduranbot_routine.pause;
		~playbutton.valueAction = 0;
		~playbutton.value = 0;
		~timeslider.valueAction = 0;
		if(~ai_assist_compose != 1, {"piece stopped".postln;});
		~fullpiecedur2 = ~fullpiecedur;
		if(~stopbutton != nil, {~stopbutton.close});
		~pause_piece = 0;
		~playbutton.string = "Play Piece";
		~duranduranbot_routine.stop;
		AppClock.sched(0.25, {
			/*~timer_routine.stop;*/
			/*Server.default.makeBundle(Server.default.latency, {~number_of_instruments.do(j = 0; {("duranduranbotsynth"++(j=j+1)).asSymbol.envirGet.set(\amp, 0);});});
			*/	/*~piece_position = 0;*/
			Server.default.makeBundle(Server.default.latency, {~number_of_instruments.do(x = 0; {("duranduranbotsynth"++(x=x+1)).asSymbol.envirGet.free;});});
			if(~will_record == 1, {
				AppClock.sched(0.5, {
					Server.default.stopRecording;
				});
			});
		});/*
		if(~num_inst_menu_num == 1, {~num_insts_menu_func.value; ~num_inst_menu_num = nil;});*/
	};

	~stopbuttonfunc = {
		~stopbutton = Button.new(~firstview, Rect(~firstviewboundsWIDTH/2.8, (x=0; ~button_base.value;), ~firstviewboundsWIDTH/8, ~firstviewboundsHEIGHT/21)).states_([["Stop piece",~stringcolor,~backgroundcolor]]).action_({arg button;
			~stop_func.value;
		});

		~stopbutton.font = Font(~default_font, size: ~firstviewboundsWIDTH/48);
	};

	if(~duranduranbot_routine.isPlaying == true, {~playbutton.value = 1; ~stopbutton.close; ~stopbuttonfunc.value}, {~playbutton.value = 0});

	~quicknewpiecebutton = Button.new(~firstview, Rect(~firstviewboundsWIDTH/72, (x=1; ~button_base.value;), ~firstviewboundsWIDTH/5.7, ~firstviewboundsHEIGHT/21)).states_([["Quick New Piece",~stringcolor,~backgroundcolor],["Quick New Piece",~stringcolor,~backgroundcolor]]).action_({arg button;
		var quicknewpiecewait;
		quicknewpiecewait = StaticText(~firstview, Rect(~firstviewboundsWIDTH/72, ~firstviewboundsHEIGHT/9.5, ~firstviewboundsWIDTH/6.2608695, ~firstviewboundsHEIGHT/22.55));
		quicknewpiecewait.background_(~backgroundcolor);
		quicknewpiecewait.stringColor_(~stringcolor);
		quicknewpiecewait.string = "please wait";
		quicknewpiecewait.font = Font(~default_font, size: ~firstviewboundsWIDTH/55);
		"please wait".postln;
		if(~piecegenerated != nil, {~piecegenerated.close});
		~quicknewpiecebutton_num = 1;
		{if(~duranduranbot_routine.isPlaying == true, {~stop_func.value});}.value;
		AppClock.sched(0, {
			if(~seedscheck == 1, {~newrandomseedsbutton.valueAction = 0}, {~seedscheck = 1});
			~generatepiecebutton.valueAction = 0; //piece fires at end of piece gen func
		});
		quicknewpiecewait.close;
	});
	~quicknewpiecebutton.font = Font(~default_font, size: ~firstviewboundsWIDTH/48);

	~quicknewpiecebutton_revert = Button.new(~firstview, Rect(~firstviewboundsWIDTH/5, (x=1; ~button_base.value;), ~firstviewboundsWIDTH/7.6, ~firstviewboundsHEIGHT/21)).states_([["Revert Piece",~stringcolor,~backgroundcolor],["Revert Piece",~stringcolor,~backgroundcolor]]).action_({arg button;
		var quicknewpiecewait;
		quicknewpiecewait = StaticText(~firstview, Rect(~firstviewboundsWIDTH/72, ~firstviewboundsHEIGHT/9.5, ~firstviewboundsWIDTH/6.2608695, ~firstviewboundsHEIGHT/22.55));
		quicknewpiecewait.background_(~stringcolor);
		quicknewpiecewait.stringColor_(~stringcolor);
		quicknewpiecewait.string = "Please Wait";
		if(~piecegenerated != nil, {~piecegenerated.close});
		quicknewpiecewait.font = Font(~default_font, size: ~firstviewboundsWIDTH/55);
		"please wait".postln;
		~quicknewpiecebutton_num = 1;

		if(~duranduranbot_routine.isPlaying == true, {~stop_func.value;});

		AppClock.sched(0, {
			if(~seedscheck == 1, {~reversenewrandomseedsbutton.valueAction = 0}, {~seedscheck = 1});
			~generatepiecebutton.valueAction = 0; //piece fires at end of piece gen func
		});
		quicknewpiecewait.close;
	});
	~quicknewpiecebutton_revert.font = Font(~default_font, size: ~firstviewboundsWIDTH/48);

	~net_change_mult_array = [1, 2, 4, 8, 16];

	~rate_button_1 = Button.new(~firstview, Rect(~firstviewboundsWIDTH/5.29, (x=2; ~button_base.value;), ~firstviewboundsWIDTH/12, ~firstviewboundsHEIGHT/21)).states_([["x1 pts",~stringcolor,~backgroundcolor], ["x1 pts",~stringcolor,Color.blue]]).action_({arg button;
		~net_change_mult = ~net_change_mult_array.at(0);
		g = File((~ddbot_root)++"/program_files/other_program_files/set_machine_learning_rate.scd", "w");
		g.write(~net_change_mult.asString);
		g.close;
		~rate_button_1.value = 1;
		~rate_button_2.value = 0;
		~rate_button_3.value = 0;
		~rate_button_4.value = 0;
		~rate_button_5.value = 0;
	});
	~rate_button_1.font = Font(~default_font, size: ~firstviewboundsWIDTH/48);

	~rate_button_2 = Button.new(~firstview, Rect(~firstviewboundsWIDTH/3.62, (x=2; ~button_base.value;), ~firstviewboundsWIDTH/12, ~firstviewboundsHEIGHT/21)).states_([["x2 pts",~stringcolor,~backgroundcolor], ["x2 pts",~stringcolor,Color.blue]]).action_({arg button;
		~net_change_mult = ~net_change_mult_array.at(1);
		g = File((~ddbot_root)++"/program_files/other_program_files/set_machine_learning_rate.scd", "w");
		g.write(~net_change_mult.asString);
		g.close;
		~rate_button_1.value = 0;
		~rate_button_2.value = 1;
		~rate_button_3.value = 0;
		~rate_button_4.value = 0;
		~rate_button_5.value = 0;
	});
	~rate_button_2.font = Font(~default_font, size: ~firstviewboundsWIDTH/48);

	~rate_button_3 = Button.new(~firstview, Rect(~firstviewboundsWIDTH/2.75, (x=2; ~button_base.value;), ~firstviewboundsWIDTH/12, ~firstviewboundsHEIGHT/21)).states_([["x4 pts",~stringcolor,~backgroundcolor], ["x4 pts",~stringcolor,Color.blue]]).action_({arg button;
		~net_change_mult = ~net_change_mult_array.at(2);
		g = File((~ddbot_root)++"/program_files/other_program_files/set_machine_learning_rate.scd", "w");
		g.write(~net_change_mult.asString);
		g.close;
		~rate_button_1.value = 0;
		~rate_button_2.value = 0;
		~rate_button_3.value = 1;
		~rate_button_4.value = 0;
		~rate_button_5.value = 0;
	});
	~rate_button_3.font = Font(~default_font, size: ~firstviewboundsWIDTH/48);

	~rate_button_4 = Button.new(~firstview, Rect(~firstviewboundsWIDTH/2.22, (x=2; ~button_base.value;), ~firstviewboundsWIDTH/12, ~firstviewboundsHEIGHT/21)).states_([["x8 pts",~stringcolor,~backgroundcolor], ["x8 pts",~stringcolor,Color.blue]]).action_({arg button;
		~net_change_mult = ~net_change_mult_array.at(3);
		g = File((~ddbot_root)++"/program_files/other_program_files/set_machine_learning_rate.scd", "w");
		g.write(~net_change_mult.asString);
		g.close;
		~rate_button_1.value = 0;
		~rate_button_2.value = 0;
		~rate_button_3.value = 0;
		~rate_button_4.value = 1;
		~rate_button_5.value = 0;
	});
	~rate_button_4.font = Font(~default_font, size: ~firstviewboundsWIDTH/48);

	~rate_button_5 = Button.new(~firstview, Rect(~firstviewboundsWIDTH/1.855, (x=2; ~button_base.value;), ~firstviewboundsWIDTH/12, ~firstviewboundsHEIGHT/21)).states_([["x16 pts",~stringcolor,~backgroundcolor], ["x16 pts",~stringcolor,Color.blue]]).action_({arg button;
		~net_change_mult = ~net_change_mult_array.at(4);
		g = File((~ddbot_root)++"/program_files/other_program_files/set_machine_learning_rate.scd", "w");
		g.write(~net_change_mult.asString);
		g.close;
		~rate_button_1.value = 0;
		~rate_button_2.value = 0;
		~rate_button_3.value = 0;
		~rate_button_4.value = 0;
		~rate_button_5.value = 1;
	});
	~rate_button_5.font = Font(~default_font, size: ~firstviewboundsWIDTH/48);

	~rate_button = Button.new(~firstview, Rect(~firstviewboundsWIDTH/72, (x=2; ~button_base.value;), ~firstviewboundsWIDTH/12, ~firstviewboundsHEIGHT/21)).states_([["Rate",~backgroundcolor,Color.green]]).action_({arg button;

		if(~duranduranbot_routine.isPlaying == true, {
			~machine_learning_func.value;
			("\nlast "++(~rate_time_range)++" secs rated:\nx"++(~net_change_mult)++" points\n").postln;
		}, {
			"pice must be running to rate".postln;
			if(~ratedonetext != nil, {~ratedonetext.close; ~ratedonetext = nil;});
			~ratedonetext = StaticText(~firstview, Rect(~firstviewboundsWIDTH/3.063829787234, ~firstviewboundsHEIGHT/4.228571428571, ~firstviewboundsWIDTH/3.3, ~firstviewboundsHEIGHT/21));
			~ratedonetext.background=~backgroundcolor;
			~ratedonetext.stringColor_(~stringcolor);
			~ratedonetext.string = "PIECE MUST BE RUNNING TO RATE";
			~ratedonetext.font = Font(~default_font, size: ~firstviewboundsWIDTH/55);

			AppClock.sched(3, {if(~ratedonetext != nil, {~ratedonetext.close; ~ratedonetext = nil;});});
		});
	});
	~rate_button.font = Font(~default_font, size: ~firstviewboundsWIDTH/48);

	if(~net_change_mult == nil, {
		case
		{~rate_button_default == ~net_change_mult_array.at(0);}{~rate_button_1.valueAction = 0;}
		{~rate_button_default == ~net_change_mult_array.at(1);}{~rate_button_2.valueAction = 0;}
		{~rate_button_default == ~net_change_mult_array.at(2);}{~rate_button_3.valueAction = 0;}
		{~rate_button_default == ~net_change_mult_array.at(3);}{~rate_button_4.valueAction = 0;}
		{~rate_button_default == ~net_change_mult_array.at(4);}{~rate_button_5.valueAction = 0;}
	}, {
		case
		{~net_change_mult == ~net_change_mult_array.at(0);}{~rate_button_1.value = 1;}
		{~net_change_mult == ~net_change_mult_array.at(1);}{~rate_button_2.value = 1;}
		{~net_change_mult == ~net_change_mult_array.at(2);}{~rate_button_3.value = 1;}
		{~net_change_mult == ~net_change_mult_array.at(3);}{~rate_button_4.value = 1;}
		{~net_change_mult == ~net_change_mult_array.at(4);}{~rate_button_5.value = 1;};
	});

	/*rate_revert_button = Button.new(~firstview, Rect(~firstviewboundsWIDTH/9.88, ~firstviewboundsHEIGHT/3.228571428571, ~firstviewboundsWIDTH/12, ~firstviewboundsHEIGHT/21)).states_([["Revert Rate",~stringcolor,~backgroundcolor]]).action_({arg button;
	~ml_revert_func.value;
	});
	rate_revert_button.font = Font(~default_font, size: ~firstviewboundsWIDTH/68);*/

	~ratepiecetext = StaticText(~firstview, Rect(~firstviewboundsWIDTH/72, (x=2.8; ~button_base.value;), ~firstviewboundsWIDTH/1.5, ~firstviewboundsHEIGHT/12));
	~ratepiecetext.background=~backgroundcolor;
	~ratepiecetext.stringColor_(~stringcolor);
	~ratepiecetext.string = "^rate last "++(~rate_time_range)++" seconds from current position in piece by clicking on a xpt rating above, and clicking "++"Rate".quote++". Higher point # = faster training, but less fine-tuning.";
	~ratepiecetext.font = Font(~default_font, size: ~firstviewboundsWIDTH/55);

	/*rangeslider = EZRanger(~firstview, bounds: (Rect(~firstviewboundsWIDTH/72, ~firstviewboundsHEIGHT/2.1951428571, ~firstviewboundsWIDTH/1.5, ~firstviewboundsHEIGHT/21)), controlSpec: ControlSpec(0, if(~piecedur_secs != nil, {~piecedur_secs}, {0}), \lin, 1, 50, "secs"), action: {
	|rangeselect|
	var rate_selection_start1;
	~rate_selection_start = (rangeselect.value.at(0)*(~instrument_1_full.size/~piecedur_secs));
	rate_selection_start1 = ~rate_selection_start.roundUp;
	if(rate_selection_start1 != ~rate_selection_start, {~rate_selection_start = rate_selection_start1-1});
	~rate_selection_end = (rangeselect.value.at(1)*(~instrument_1_full.size/~piecedur_secs)).roundUp;
	/*rangeselect.value.postln;*/

	}, initVal: [0,if(~piecedur_secs != nil, {~piecedur_secs}, {0})]);
	rangeslider.setColors(~stringcolor,~stringcolor, ~stringcolor(0.7),~stringcolor, ~stringcolor, Color.yellow);
	rangeslider.font = Font(~default_font, size: ~firstviewboundsWIDTH/55);*/

	/*rate_selectionstatictext = StaticText(~firstview, Rect(~firstviewboundsWIDTH/72, ~firstviewboundsHEIGHT/1.502222222222 /*~firstviewboundsHEIGHT/6.24286*/, ~firstviewboundsWIDTH/2.7, ~firstviewboundsHEIGHT/21));
	rate_selectionstatictext.background=~stringcolor;
	rate_selectionstatictext.string = if(~fullpiecedur != nil, {~fullpiecedur}, {"piece length - --:-- mins"});
	rate_selectionstatictext.font = Font(~default_font, size: ~firstviewboundsWIDTH/48);*/

	if(~piece_position == nil, {~piece_position = 0});

	~timeslider_func = {
		~timeslider = EZSlider(~firstview,  bounds: (Rect(~firstviewboundsWIDTH/72, if(~ai_assist_compose == 0, {(x=4.3; ~button_base.value;)}, {~firstviewboundsHEIGHT/5}), ~firstviewboundsWIDTH/1.5, ~firstviewboundsHEIGHT/21)), controlSpec: ControlSpec(0, ("instrument_"++(1)++"_full").asSymbol.envirGet.size - 1, \lin, 1, 0, "bits"), action: {
			|timeselect|
			var selection1;
			~piece_position = timeselect.value/**(~instrument_1_full.size/~piecedur_secs)*/; // works
			/*selection1 = ~piece_position.roundUp;
			if(selection1 != ~piece_position, {~piece_position = selection1-1});*/
			/*if(timeslider.value >= ~rate_time_range, {rangeslider.valueAction = [timeslider.value-~rate_time_range, timeslider.value]}, {rangeslider.valueAction = [0, timeslider.value]});
			*//*timeselect.value.postln;
			~piece_position.postln;*/
			/*if((~piecedur_secs_zero+1) != timeselect.value, {
			~piecedur_secs_zero = timeselect.value;
			~fullpiecedur_change = ((~piecedur_secs - ~piecedur_secs_zero)/60);
			~dd_secs = ((~fullpiecedur_change-(~dd_mins = ~fullpiecedur_change.roundUp-1))*60).roundUp;
			});
			*/
		});
		~timeslider.setColors(~stringcolor,~stringcolor);

	};

	~timeslider_func.value;

	/*if(~pause_piece != nil, {if(~pause_piece != 1, {~timeslider.valueAction = 0;/*rangeslider.valueAction = [0, if(~piecedur_secs != nil, {~piecedur_secs}, {0});]*/nil});});
	*/
	/*if(~pause_piece == 0, {~piece_position = 0});*/

	~timeslidertext = StaticText(~firstview, Rect(~firstviewboundsWIDTH/72, (x=5.2; ~button_base.value;), ~firstviewboundsWIDTH/1.5, ~firstviewboundsHEIGHT/22.55));
	~timeslidertext.background=~backgroundcolor;
	~timeslidertext.stringColor=~stringcolor;
	~timeslidertext.string = "^Jump to location in current piece by using the slider just above";
	~timeslidertext.font = Font(~default_font, size: ~firstviewboundsWIDTH/55);

	~piecedurationstatictext = StaticText(~firstview, Rect(~firstviewboundsWIDTH/72, (x=6; ~button_base.value;) /*~firstviewboundsHEIGHT/6.24286*/, ~firstviewboundsWIDTH/2.7, ~firstviewboundsHEIGHT/21));
	~piecedurationstatictext.background=~backgroundcolor;
	~piecedurationstatictext.stringColor=~stringcolor;
	~piecedurationstatictext.string = if(~fullpiecedur != nil, {~fullpiecedur}, {"Piece Length - --:-- mins"});
	~piecedurationstatictext.font = Font(~default_font, size: ~firstviewboundsWIDTH/48);

	~piecepositionstatictext = StaticText(~firstview, Rect((~firstviewboundsWIDTH/72)+(~firstviewboundsWIDTH/2.95)+30, (x=6; ~button_base.value;) /*~firstviewboundsHEIGHT/6.24286*/, ~firstviewboundsWIDTH/2.7, ~firstviewboundsHEIGHT/21));
	~piecepositionstatictext.background=~backgroundcolor;
	~piecepositionstatictext.stringColor=~stringcolor;
	~piecepositionstatictext.string = if(~instrument_1_full_midi != nil, {(
		"Bit pos. "++(~piece_position.asInteger.asString)++" Midi "++(
			case
			{~number_of_instruments == 1}{(if(~instrument_1_full_midi != nil, {~instrument_1_full_midi.at(~piece_position).asString}))}
			{~number_of_instruments == 2}{(if(~instrument_1_full_midi != nil, {~instrument_1_full_midi.at(~piece_position).asString}))++" "++(if(~instrument_2_full_midi != nil, {~instrument_2_full_midi.at(~piece_position).asString}))}
			{~number_of_instruments == 3}{(if(~instrument_1_full_midi != nil, {~instrument_1_full_midi.at(~piece_position).asString}))++" "++(if(~instrument_2_full_midi != nil, {~instrument_2_full_midi.at(~piece_position).asString}))++" "++(if(~instrument_3_full_midi != nil, {~instrument_3_full_midi.at(~piece_position).asString}))};
		)
	)}, {"Bit pos. -- Midi --"});
	~piecepositionstatictext.font = Font(~default_font, size: ~firstviewboundsWIDTH/48);

	if(~pos_1 == nil, {~pos_1 = 0});
	if(~pos_2 == nil, {~pos_2 = 0});

	~loopstatictext_1 = StaticText(~firstview, Rect(~firstviewboundsWIDTH/8.5, (x=7; ~button_base.value;), ~firstviewboundsWIDTH/12, ~firstviewboundsHEIGHT/21));
	~loopstatictext_1.background_(~backgroundcolor);
	~loopstatictext_1.stringColor_(~stringcolor);
	~loopstatictext_1.string = "Bit pos. 1";
	~loopstatictext_1.font = Font(~default_font, size: ~firstviewboundsWIDTH/55);

	~looptextfield_1 = TextField(~firstview, Rect(~firstviewboundsWIDTH/4.9, (x=7; ~button_base.value;), ~firstviewboundsWIDTH/10, ~firstviewboundsHEIGHT/21));
	~looptextfield_1.string = ~pos_1.asString;
	~looptextfield_1.background_(~backgroundcolor);
	~looptextfield_1.stringColor_(~stringcolor);
	~looptextfield_1.font = Font(~default_font, size: ~firstviewboundsWIDTH/48);
	~looptextfield_1.action = {
		~sdfk = ~looptextfield_1.value.asInteger;
		/*	~pos_1 = looptextfield_1.value.asInteger;*/
		if(~sdfk.value.isInteger == true, {~pos_1 = ~sdfk.value.asInteger;}, {~looptextfield_1.string = "integer".asString;});
		~looptextfield_1.string = ~pos_1.asString;
	};

	~loopstatictext_2 = StaticText(~firstview, Rect(~firstviewboundsWIDTH/3.2, (x=7; ~button_base.value;), ~firstviewboundsWIDTH/12, ~firstviewboundsHEIGHT/21));
	~loopstatictext_2.background_(~backgroundcolor);
	~loopstatictext_2.stringColor_(~stringcolor);
	~loopstatictext_2.string = "Bit pos. 2";
	~loopstatictext_2.font = Font(~default_font, size: ~firstviewboundsWIDTH/55);

	~looptextfield_2 = TextField(~firstview, Rect(~firstviewboundsWIDTH/2.5, (x=7; ~button_base.value;), ~firstviewboundsWIDTH/10, ~firstviewboundsHEIGHT/21));
	~looptextfield_2.string = ~pos_2.asString;
	~looptextfield_2.background_(~backgroundcolor);
	~looptextfield_2.stringColor_(~stringcolor);
	~looptextfield_2.font = Font(~default_font, size: ~firstviewboundsWIDTH/48);
	~looptextfield_2.action = {
		~sdfk = ~looptextfield_2.value.asInteger;
		if(~sdfk.value.isInteger == true, {~pos_2 = ~sdfk.value.asInteger;}, {~looptextfield_2.string = "integer".asString;});
		~looptextfield_2.string = ~pos_2.asString;
	};

	/*~loop_creator_func = {

	if(("instrument_"++(1)++"_full_midi_before").asSymbol.envirGet == nil, {
	~number_of_instruments.do(t = 0; {
	t=t+1;
	("instrument_"++(t)++"_full_midi_before").asSymbol.envirPut(("instrument_"++(t)++"_full_midi").asSymbol.envirGet;);
	});
	});

	(
	h = 0.midicps;
	z = ~pos_1 - ~pos_2;
	if(z.isNegative == true, {z = z.neg;});
	z=z+1;
	~number_of_instruments.do(t = 0; {
	t=t+1;
	("loop_new_array"++(t)).asSymbol.envirPut([]);
	z.do(q = ~pos_1; {
	("loop_new_array"++(t)).asSymbol.envirPut(
	("loop_new_array"++(t)).asSymbol.envirGet.add(("instrument_"++(t)++"_full_midi").asSymbol.envirGet.at(q););

	);
	q=q+1;
	});

	("instrument_"++(t)++"_full_midi").asSymbol.envirPut(("loop_new_array"++(t)).asSymbol.envirGet;);

	});
	);

	~saved_piece_loaded = 1;
	~create_pseq_arrays_func.value;

	};

	/*~instrument_2_full_midi.size
	~instrument_2_full_midi.at(167)
	~instrument_2_full.at(167)
	~instrument_2_full.size*/

	loopsetbutton =  Button.new(~firstview, Rect(~firstviewboundsWIDTH/4.3, ~firstviewboundsHEIGHT/1.47, ~firstviewboundsWIDTH/10, ~firstviewboundsHEIGHT/21)).states_([["Set Loop",~stringcolor,~backgroundcolor]]).action_({arg button;
	~loop_on = 1;
	if(~duranduranbot_routine.isPlaying == true, {~stop_func.value;});
	~loop_creator_func.value;
	});

	stoploopbutton =  Button.new(~firstview, Rect(~firstviewboundsWIDTH/2.9, ~firstviewboundsHEIGHT/1.47, ~firstviewboundsWIDTH/10, ~firstviewboundsHEIGHT/21)).states_([["Stop Loop",~stringcolor,~backgroundcolor]]).action_({arg button;
	~loop_on = 0;
	~number_of_instruments.do(t = 0; {
	t=t+1;
	("instrument_"++(t)++"_full_midi").asSymbol.envirPut(("instrument_"++(t)++"_full_midi_before").asSymbol.envirGet;);
	("instrument_"++(t)++"_full_midi_before").asSymbol.envirPut(nil);
	});

	if(~duranduranbot_routine.isPlaying == true, {~stop_func.value;});
	~saved_piece_loaded = 1;
	~create_pseq_arrays_func.value;
	});*/

	if(~loop_on == nil, {~loop_on = 0});

	~loopsetbutton =  Button.new(~firstview, Rect(~firstviewboundsWIDTH/72, (x=7; ~button_base.value;), ~firstviewboundsWIDTH/10, ~firstviewboundsHEIGHT/21)).states_([["Loop is OFF",~stringcolor,~backgroundcolor],["Loop is ON",~stringcolor,~backgroundcolor]]).action_({arg button;

		if(button.value == 1, {~loop_on = 1;}, {~loop_on = 0;});
		/*if(~duranduranbot_routine.isPlaying == true, {~stop_func.value;});
		~loop_creator_func.value;*/
	});
	~loopsetbutton.font = Font(~default_font, size: ~firstviewboundsWIDTH/65);

	if(~loop_on == 1, {~loopsetbutton.value = 1}, {~loopsetbutton.value = 0});

	if(~num_loops == nil, {~num_loops = 24});

	~loopstatictext_2 = StaticText(~firstview, Rect(~firstviewboundsWIDTH/1.95, (x=7; ~button_base.value;), ~firstviewboundsWIDTH/15, ~firstviewboundsHEIGHT/21));
	~loopstatictext_2.background_(~backgroundcolor);
	~loopstatictext_2.stringColor_(~stringcolor);
	~loopstatictext_2.string = "# loops";
	~loopstatictext_2.font = Font(~default_font, size: ~firstviewboundsWIDTH/55);

	~looptextfield_3 = TextField(~firstview, Rect(~firstviewboundsWIDTH/1.72,(x=7; ~button_base.value;), ~firstviewboundsWIDTH/10, ~firstviewboundsHEIGHT/21));
	~looptextfield_3.string = ~num_loops.asString;
	~looptextfield_3.background_(~backgroundcolor);
	~looptextfield_3.stringColor_(~stringcolor);
	~looptextfield_3.font = Font(~default_font, size: ~firstviewboundsWIDTH/48);
	~looptextfield_3.action = {
		~sdfk = ~looptextfield_3.value.asInteger;
		if(~sdfk.value.isInteger == true, {~num_loops = ~sdfk.value.asInteger;}, {~looptextfield_3.string = "integer".asString;});
		~looptextfield_3.string = ~num_loops.asString;
	};

	~recordbutton = Button.new(~firstview, Rect(~firstviewboundsWIDTH/72, (x=7.8; ~button_base.value;), ~firstviewboundsWIDTH/9, ~firstviewboundsHEIGHT/21)).states_([["Record Off",~stringcolor,~backgroundcolor],["Record On",~backgroundcolor,Color.red]]).action_({arg button;
		if(button.value == 1, {
			~recordtext.string = "All pieces WILL be recorded";
			~will_record = 1;
		}, {~recordtext.string = "All pieces will NOT be recorded"; ~will_record = 0;});

	});
	~recordbutton.font = Font(~default_font, size: ~firstviewboundsWIDTH/48);

	~recordtext = StaticText(~firstview, Rect(~firstviewboundsWIDTH/7.3, (x=7.8; ~button_base.value;), ~firstviewboundsWIDTH/2.2, ~firstviewboundsHEIGHT/21));
	~recordtext.background=~backgroundcolor;
	~recordtext.stringColor=~stringcolor;
	~recordtext.string = "All pieces will NOT be recorded";
	~recordtext.font = Font(~default_font, size: ~firstviewboundsWIDTH/55);

	if(~will_record == 1, {~recordtext.string = "All pieces WILL be recorded"; ~recordbutton.value = 1;},
		{~recordtext.string = "All pieces will NOT be recorded"; ~recordbutton.value = 0;});

	/*ctrlperiodtext = StaticText(~firstview, Rect(~firstviewboundsWIDTH/1.4173228346457, ~firstviewboundsHEIGHT/13.5, ~firstviewboundsWIDTH/3.5, ~firstviewboundsHEIGHT/4.37));
	ctrlperiodtext.background=~backgroundcolor;
	ctrlperiodtext.string = "*Important Note*... You can kill all processes by hitting "++"cmd/ctrl - period".quote++"on your keyboard.";
	ctrlperiodtext.align = \topLeft;
	ctrlperiodtext.font = Font(~default_font, size: ~firstviewboundsWIDTH/70.454545454545);*/

	~newpieceautobutton = Button.new(~firstview, Rect(~firstviewboundsWIDTH/72, (x=8.6; ~button_base.value;), ~firstviewboundsWIDTH/5, ~firstviewboundsHEIGHT/21)).states_([["Auto New Piece On",~stringcolor,~backgroundcolor],["Auto New Piece Off",~stringcolor,~backgroundcolor]]).action_({arg button;
		if(button.value == 1, {
			g = File((~ddbot_root)++"/program_files/other_program_files/autonewpiece.scd", "w");
			g.write("0");
			g.close;
			~newpieceautotext.string = "Automatic "++"Quick New Piece".quote++" is OFF";
		}, {
			~newpieceautotext.string = "Automatic "++"Quick New Piece".quote++" is ON";
			g = File((~ddbot_root)++"/program_files/other_program_files/autonewpiece.scd", "w");
			g.write("1");
			g.close;
		});

	});
	~newpieceautobutton.font = Font(~default_font, size: ~firstviewboundsWIDTH/48);

	~newpieceautotext = StaticText(~firstview, Rect(~firstviewboundsWIDTH/4.5, (x=8.6; ~button_base.value;), ~firstviewboundsWIDTH/2.2, ~firstviewboundsHEIGHT/21));
	~newpieceautotext.background=~backgroundcolor;
	~newpieceautotext.stringColor=~stringcolor;
	if((FileReader.read((~ddbot_root)++"/program_files/other_program_files/autonewpiece.scd", false, true).flatten.at(0) == 1.asString), {
		~newpieceautobutton.value = 0; ~newpieceautotext.string = "Automatic "++"Quick New Piece".quote++" is ON";
	}, {~newpieceautobutton.value = 1; ~newpieceautotext.string = "Automatic "++"Quick New Piece".quote++" is OFF";});
	~newpieceautotext.font = Font(~default_font, size: ~firstviewboundsWIDTH/55);

	~loadscorebuttonbutton = Button.new(~firstview, Rect(~firstviewboundsWIDTH/72, (x=9.4; ~button_base.value;), ~firstviewboundsWIDTH/5.4, ~firstviewboundsHEIGHT/21)).states_([["Load Saved Score",~stringcolor,~backgroundcolor]]).action_({arg button;

		FileDialog.new(
			{arg path;
				File.readAllString(path, r).interpret.value;
				~saved_piece_loaded = 1;
				if(~duranduranbot_routine.isPlaying == true, {~stop_func.value;});
				~generatepiecebutton.valueAction = 1;
				~loadscoretext.string = "Score has been loaded. Press "++"Play Piece".quote++" to run.";
				AppClock.sched(5, {~loadscoretext.string = "Load a saved score here. Then press "++"Play Piece".quote++".";});
		}, {}, 0, 0, stripResult: true, path: ((~ddbot_root)++"/program_files/saved_scores"));
	});
	~loadscorebuttonbutton.font = Font(~default_font, size: ~firstviewboundsWIDTH/48);

	~loadscoretext = StaticText(~firstview, Rect(~firstviewboundsWIDTH/4.9, (x=9.4; ~button_base.value;), ~firstviewboundsWIDTH/1.3, ~firstviewboundsHEIGHT/21));
	~loadscoretext.background=~backgroundcolor;
	~loadscoretext.stringColor=~stringcolor;
	~loadscoretext.string = "Load a saved score here. Then press "++"Play Piece".quote++".";
	~loadscoretext.font = Font(~default_font, size: ~firstviewboundsWIDTH/55);

	~volbutton_func = {

		~volbutton = Button.new(~firstview,
			if(~ai_assist_compose != 1, {
				Rect(~firstviewboundsWIDTH/1.4173228346457, ~firstviewboundsHEIGHT/6.24286, ~firstviewboundsWIDTH/18, ~firstviewboundsHEIGHT/21)
			}, {
				Rect(~firstviewboundsWIDTH/1.37, (x=0; ~button_base.value;), ~firstviewboundsWIDTH/18, ~firstviewboundsHEIGHT/21)
			})
		).states_([["Vol.",~stringcolor,~backgroundcolor],["Vol.",~backgroundcolor,Color.green]]).action_({arg button;

			var amp_before_1, amp_before_2, amp_before_3;

			if(~volbutton.value == 1, {

				~volview = View(~firstview,
					if(~ai_assist_compose != 1, {
						Rect(0, ~firstviewboundsHEIGHT/4.4, ~firstviewboundsWIDTH/3.56*4, ~firstviewboundsHEIGHT/2)
					}, {
						Rect(0, (x=1; ~button_base.value;), ~firstviewboundsWIDTH/3.8*4, ~firstviewboundsHEIGHT/2)
					})
				);
				~volview.background_(~backgroundcolor);

				amp_before_1 = ~amp_1;
				amp_before_2 = ~amp_2;
				amp_before_3 = ~amp_3;

				~volslider1 = EZSlider(~volview,  bounds: (Rect(~firstviewboundsWIDTH/72, 0, ~firstviewboundsWIDTH/40, ~firstviewboundsHEIGHT/2)), controlSpec: ControlSpec(0, 4, \lin, 0.1, ~amp_1), action: {
					|timeselect|
					var selection1;
					~amp_1 = timeselect.value;
					~voltext1.string = "i1 "++(~amp_1.asString);
					/*~trimviewtext3.string = "Keep from bit "++(~cut_from)++" to "++(~cut_to)++", and discard the rest.";*/
				}, layout: 'vert');
				~volslider1.setColors(~stringcolor,~stringcolor);

				~volinsttext1 = StaticText(~volview, Rect(~firstviewboundsWIDTH/72 + (~firstviewboundsWIDTH/40*1.5), 0, ~firstviewboundsWIDTH/5.5, ~firstviewboundsHEIGHT/21));
				~volinsttext1.background_(~backgroundcolor);
				~volinsttext1.stringColor_(~stringcolor);
				~volinsttext1.string = "Instrument 1";
				~volinsttext1.font = Font(~default_font, size: ~firstviewboundsWIDTH/55);

				~defaultvolbutton1 = Button.new(~volview, Rect(~firstviewboundsWIDTH/72 + (~firstviewboundsWIDTH/40*1.5), ~firstviewboundsHEIGHT/21*1.5, ~firstviewboundsWIDTH/6, ~firstviewboundsHEIGHT/21)).states_([[("set to default - "++(~amp_default_1.asString)),~stringcolor,~backgroundcolor]]).action_({arg button;

					~amp_1 = ~amp_default_1;
					~volslider1.value = ~amp_1;
					~voltext1.string = "i1 "++(~amp_1.asString);
					if(amp_before_1 != ~amp_1, {
						g = File((~ddbot_root)++"/program_files/other_program_files/amplitude_1.scd", "w");
						g.write(~amp_1.asString;);
						g.close;
					});
				});
				~defaultvolbutton1.font = Font(~default_font, size: ~firstviewboundsWIDTH/55);

				if(~number_of_instruments > 1, {

					~volslider2 = EZSlider(~volview,  bounds: (Rect(~firstviewboundsWIDTH/72*2 + (((~firstviewboundsWIDTH/40) + (~firstviewboundsWIDTH/5.5))*1), 0, ~firstviewboundsWIDTH/40, ~firstviewboundsHEIGHT/2)), controlSpec: ControlSpec(0, 4, \lin, 0.1, ~amp_2), action: {
						|timeselect|
						var selection1;
						~amp_2 = timeselect.value;
						~voltext2.string = "i2 "++(~amp_2.asString);
						/*~trimviewtext3.string = "Keep from bit "++(~cut_from)++" to "++(~cut_to)++", and discard the rest.";*/
					}, layout: 'vert');
					~volslider2.setColors(~stringcolor,~stringcolor);

					~volinsttext2 = StaticText(~volview, Rect(~firstviewboundsWIDTH/72*2 + (((~firstviewboundsWIDTH/40) + (~firstviewboundsWIDTH/5.5))*1) + (~firstviewboundsWIDTH/40*1.5), 0, ~firstviewboundsWIDTH/5.5, ~firstviewboundsHEIGHT/21));
					~volinsttext2.background_(~backgroundcolor);
					~volinsttext2.stringColor_(~stringcolor);
					~volinsttext2.string = "Instrument 2";
					~volinsttext2.font = Font(~default_font, size: ~firstviewboundsWIDTH/55);

					~defaultvolbutton2 = Button.new(~volview, Rect(~firstviewboundsWIDTH/72*2 + (((~firstviewboundsWIDTH/40) + (~firstviewboundsWIDTH/5.5))*1) + (~firstviewboundsWIDTH/40*1.5), ~firstviewboundsHEIGHT/21*1.5, ~firstviewboundsWIDTH/6, ~firstviewboundsHEIGHT/21)).states_([[("set to default - "++(~amp_default_2.asString)),~stringcolor,~backgroundcolor]]).action_({arg button;

						~amp_2 = ~amp_default_2;
						~volslider2.value = ~amp_2;
						~voltext2.string = "i2 "++(~amp_2.asString);
						if(amp_before_2 != ~amp_2, {
							g = File((~ddbot_root)++"/program_files/other_program_files/amplitude_2.scd", "w");
							g.write(~amp_2.asString;);
							g.close;
						});
					});
					~defaultvolbutton2.font = Font(~default_font, size: ~firstviewboundsWIDTH/55);

				});

				if(~number_of_instruments > 2, {

					~volslider3 = EZSlider(~volview,  bounds: (Rect(~firstviewboundsWIDTH/72*3 + (((~firstviewboundsWIDTH/40) + (~firstviewboundsWIDTH/5.5))*2), 0, ~firstviewboundsWIDTH/40, ~firstviewboundsHEIGHT/2)), controlSpec: ControlSpec(0, 4, \lin, 0.1, ~amp_3), action: {
						|timeselect|
						var selection1;
						~amp_3 = timeselect.value;
						~voltext3.string = "i3 "++(~amp_3.asString);
						/*~trimviewtext3.string = "Keep from bit "++(~cut_from)++" to "++(~cut_to)++", and discard the rest.";*/
					}, layout: 'vert');
					~volslider3.setColors(~stringcolor,~stringcolor);

					~volinsttext3 = StaticText(~volview, Rect(~firstviewboundsWIDTH/72*3 + (((~firstviewboundsWIDTH/40) + (~firstviewboundsWIDTH/5.5))*2) + (~firstviewboundsWIDTH/40*1.5), 0, ~firstviewboundsWIDTH/5.5, ~firstviewboundsHEIGHT/21));
					~volinsttext3.background_(~backgroundcolor);
					~volinsttext3.stringColor_(~stringcolor);
					~volinsttext3.string = "Instrument 3";
					~volinsttext3.font = Font(~default_font, size: ~firstviewboundsWIDTH/55);

					~defaultvolbutton3 = Button.new(~volview, Rect(~firstviewboundsWIDTH/72*3 + (((~firstviewboundsWIDTH/40) + (~firstviewboundsWIDTH/5.5))*2) + (~firstviewboundsWIDTH/40*1.5), ~firstviewboundsHEIGHT/21*1.5, ~firstviewboundsWIDTH/6, ~firstviewboundsHEIGHT/21)).states_([[("set to default - "++(~amp_default_3.asString)),~stringcolor,~backgroundcolor]]).action_({arg button;

						~amp_3 = ~amp_default_3;
						~volslider3.value = ~amp_3;
						~voltext3.string = "i3 "++(~amp_3.asString);
						if(amp_before_3 != ~amp_3, {
							g = File((~ddbot_root)++"/program_files/other_program_files/amplitude_3.scd", "w");
							g.write(~amp_3.asString;);
							g.close;
						});
					});
					~defaultvolbutton3.font = Font(~default_font, size: ~firstviewboundsWIDTH/55);

				});

				if(~number_of_instruments > 1, {

					~volslider0 = EZSlider(~volview,  bounds: (Rect(~firstviewboundsWIDTH/72*4 + (((~firstviewboundsWIDTH/40) + (~firstviewboundsWIDTH/5.5))*3),0, ~firstviewboundsWIDTH/40, ~firstviewboundsHEIGHT/2)), controlSpec: ControlSpec(0, 4, \lin, 0.1, ~amp_1), action: {
						|timeselect|
						var selection1;
						~amp_1 = timeselect.value;
						~voltext1.string = "i1 "++(~amp_1.asString);
						if(~number_of_instruments > 1, {
							~amp_2 = timeselect.value;
							~voltext2.string = "i2 "++(~amp_2.asString);
						});
						if(~number_of_instruments > 2, {
							~amp_3 = timeselect.value;
							~voltext3.string = "i3 "++(~amp_3.asString);
						});
						/*~trimviewtext3.string = "Keep from bit "++(~cut_from)++" to "++(~cut_to)++", and discard the rest.";*/
					}, layout: 'vert');
					~volslider0.setColors(~stringcolor,~stringcolor);

					~volinsttext0 = StaticText(~volview, Rect(~firstviewboundsWIDTH/72*4 + (((~firstviewboundsWIDTH/40) + (~firstviewboundsWIDTH/5.5))*3) + (~firstviewboundsWIDTH/40*1.5), 0, ~firstviewboundsWIDTH/5.5, ~firstviewboundsHEIGHT/21));
					~volinsttext0.background_(~backgroundcolor);
					~volinsttext0.stringColor_(~stringcolor);
					~volinsttext0.string = "All Instruments";
					~volinsttext0.font = Font(~default_font, size: ~firstviewboundsWIDTH/55);

					~defaultvolbutton0 = Button.new(~volview, Rect(~firstviewboundsWIDTH/72*4 + (((~firstviewboundsWIDTH/40) + (~firstviewboundsWIDTH/5.5))*3) + (~firstviewboundsWIDTH/40*1.5), ~firstviewboundsHEIGHT/21*1.5, ~firstviewboundsWIDTH/6, ~firstviewboundsHEIGHT/21)).states_([[("set to default - "++(~amp_default_1.asString)),~stringcolor,~backgroundcolor]]).action_({arg button;

						~volslider0.value = ~amp_default_1;

						~amp_1 = ~amp_default_1;
						~volslider1.value = ~amp_1;
						~voltext1.string = "i1 "++(~amp_1.asString);
						if(~number_of_instruments > 1, {
							~amp_2 = ~amp_default_2;
							~volslider2.value = ~amp_2;
							~voltext2.string = "i2 "++(~amp_2.asString);
						});
						if(~number_of_instruments > 2, {
							~amp_3 = ~amp_default_3;
							~volslider3.value = ~amp_3;
							~voltext3.string = "i3 "++(~amp_3.asString);
						});
						if(amp_before_1 != ~amp_1, {
							g = File((~ddbot_root)++"/program_files/other_program_files/amplitude_1.scd", "w");
							g.write(~amp_1.asString;);
							g.close;
						});
						if(amp_before_2 != ~amp_2, {
							g = File((~ddbot_root)++"/program_files/other_program_files/amplitude_2.scd", "w");
							g.write(~amp_2.asString;);
							g.close;
						});
						if(amp_before_3 != ~amp_3, {
							g = File((~ddbot_root)++"/program_files/other_program_files/amplitude_3.scd", "w");
							g.write(~amp_3.asString;);
							g.close;
						});
					});
					~defaultvolbutton0.font = Font(~default_font, size: ~firstviewboundsWIDTH/55);

				});

			}, {
				if(amp_before_1 != ~amp_1, {
					g = File((~ddbot_root)++"/program_files/other_program_files/amplitude_1.scd", "w");
					g.write(~amp_1.asString;);
					g.close;
				});
				if(amp_before_2 != ~amp_2, {
					g = File((~ddbot_root)++"/program_files/other_program_files/amplitude_2.scd", "w");
					g.write(~amp_2.asString;);
					g.close;
				});
				if(amp_before_3 != ~amp_3, {
					g = File((~ddbot_root)++"/program_files/other_program_files/amplitude_3.scd", "w");
					g.write(~amp_3.asString;);
					g.close;
				});

				~volview.close;
			});
		});
		~volbutton.font = Font(~default_font, size: ~firstviewboundsWIDTH/55);

		~voltext1 = StaticText(~firstview,
			if(~ai_assist_compose != 1, {
				Rect(~firstviewboundsWIDTH/1.4173228346457 + (~firstviewboundsWIDTH/18) + (~firstviewboundsWIDTH/72), ~firstviewboundsHEIGHT/6.24286, ~firstviewboundsWIDTH/17, ~firstviewboundsHEIGHT/21)
			}, {
				Rect(~firstviewboundsWIDTH/1.37 + (~firstviewboundsWIDTH/18) + (~firstviewboundsWIDTH/72), (x=0; ~button_base.value;), ~firstviewboundsWIDTH/17, ~firstviewboundsHEIGHT/21)
			})
		);
		~voltext1.background_(~backgroundcolor);
		~voltext1.stringColor_(~stringcolor);
		~voltext1.string = "i1 "++(~amp_1.asString);
		~voltext1.font = Font(~default_font, size: ~firstviewboundsWIDTH/55);

		if(~number_of_instruments > 1, {

			~voltext2 = StaticText(~firstview,
				if(~ai_assist_compose != 1, {
					Rect(~firstviewboundsWIDTH/1.4173228346457 + (~firstviewboundsWIDTH/18) + (~firstviewboundsWIDTH/72) + (~firstviewboundsWIDTH/17), ~firstviewboundsHEIGHT/6.24286, ~firstviewboundsWIDTH/17, ~firstviewboundsHEIGHT/21)
				}, {
					Rect(~firstviewboundsWIDTH/1.37 + (~firstviewboundsWIDTH/18) + (~firstviewboundsWIDTH/72) + (~firstviewboundsWIDTH/17), (x=0; ~button_base.value;), ~firstviewboundsWIDTH/17, ~firstviewboundsHEIGHT/21)
				})
			);
			~voltext2.background_(~backgroundcolor);
			~voltext2.stringColor_(~stringcolor);
			~voltext2.string = "i2 "++(~amp_2.asString);
			~voltext2.font = Font(~default_font, size: ~firstviewboundsWIDTH/55);

		});

		if(~number_of_instruments > 2, {

			~voltext3 = StaticText(~firstview,
				if(~ai_assist_compose != 1, {
					Rect(~firstviewboundsWIDTH/1.4173228346457 + (~firstviewboundsWIDTH/18) + (~firstviewboundsWIDTH/72) + (~firstviewboundsWIDTH/17*2), ~firstviewboundsHEIGHT/6.24286, ~firstviewboundsWIDTH/17, ~firstviewboundsHEIGHT/21)
				}, {
					Rect(~firstviewboundsWIDTH/1.37 + (~firstviewboundsWIDTH/18) + (~firstviewboundsWIDTH/72) + (~firstviewboundsWIDTH/17*2), (x=0; ~button_base.value;), ~firstviewboundsWIDTH/17, ~firstviewboundsHEIGHT/21)
				})
			);
			~voltext3.background_(~backgroundcolor);
			~voltext3.stringColor_(~stringcolor);
			~voltext3.string = "i3 "++(~amp_3.asString);
			~voltext3.font = Font(~default_font, size: ~firstviewboundsWIDTH/55);

		});

	};

	~volbutton_func.value;

	~newrandomseedsbutton = Button.new(~firstview, Rect(~firstviewboundsWIDTH/1.4173228346457, ~firstviewboundsHEIGHT/4.4, ~firstviewboundsWIDTH/9, ~firstviewboundsHEIGHT/21)).states_([["New Seeds",~stringcolor,~backgroundcolor],["New Seeds",~stringcolor,~backgroundcolor]]).action_({arg button;
		~randseed_1_backup = ~randseed_1;
		~randseed_2_backup = ~randseed_2;
		~randseed_3_backup = ~randseed_3;
		~randseed_1 =  99999999.rand;
		~randseed_2 =  99999999.rand;
		~randseed_3 =  99999999.rand;
		~save_random_seeds_func.value;
		z = "Current random seeds:\n"++
		"randseed 1 - "++~randseed_1++"\n"++
		"randseed 2 - "++~randseed_2++"\n"++
		"randseed 3 - "++~randseed_3.asString++(if(~quicknewpiecebutton_num != 1, {
			"\n(also posted in post window)"++"\n"++
			"*make sure to hit "++"generate piece".quote++" to generate a new piece based on the new seeds"}, {""}));
		z.postln;
		~currentrandomseedstrext.string = z;
		~currentrandomseedstrext.background=~backgroundcolor;
		~currentrandomseedstrext.stringColor=~stringcolor;

	});
	~newrandomseedsbutton.font = Font(~default_font, size: ~firstviewboundsWIDTH/55);

	~reversenewrandomseedsbutton = Button.new(~firstview, Rect(~firstviewboundsWIDTH/1.209907834101, ~firstviewboundsHEIGHT/4.4, ~firstviewboundsWIDTH/15, ~firstviewboundsHEIGHT/21)).states_([["Revert",~stringcolor,~backgroundcolor],["Revert",~stringcolor,~backgroundcolor]]).action_({arg button;
		~randseed_1_backup2 = ~randseed_1;
		~randseed_2_backup2 = ~randseed_2;
		~randseed_3_backup2 = ~randseed_3;
		~randseed_1 =  ~randseed_1_backup;
		~randseed_2 =  ~randseed_2_backup;
		~randseed_3 =  ~randseed_3_backup;
		~randseed_1_backup = ~randseed_1_backup2;
		~randseed_2_backup = ~randseed_2_backup2;
		~randseed_3_backup = ~randseed_3_backup2;
		~save_random_seeds_func.value;
		z = "Current random seeds:\n"++
		"randseed 1 - "++~randseed_1++"\n"++
		"randseed 2 - "++~randseed_2++"\n"++
		"randseed 3 - "++~randseed_3.asString++(if(~quicknewpiecebutton_num != 1, {
			"\n(also posted in post window)"++"\n"++
			"*make sure to hit "++"generate piece".quote++" to generate a new piece based on the new seeds"}, {""}));
		z.postln;
		~currentrandomseedstrext.string = z;
		~currentrandomseedstrext.background=~backgroundcolor;
		~currentrandomseedstrext.stringColor=~stringcolor;
	});
	~reversenewrandomseedsbutton.font = Font(~default_font, size: ~firstviewboundsWIDTH/55);

	~setnewrandomseedsbutton = Button.new(~firstview, Rect(~firstviewboundsWIDTH/1.1059907834101, ~firstviewboundsHEIGHT/4.4, ~firstviewboundsWIDTH/12, ~firstviewboundsHEIGHT/21)).states_([["Set Seeds",~stringcolor,~backgroundcolor]]).action_({arg button;
		~firstview.close;
		~set_seeds_page_func.value;
	});
	~setnewrandomseedsbutton.font = Font(~default_font, size: ~firstviewboundsWIDTH/55);

	~currentrandomseedstrext = TextView(~firstview, Rect(~firstviewboundsWIDTH/1.4173228346457, ~firstviewboundsHEIGHT/3.5, ~firstviewboundsWIDTH/3.56, ~firstviewboundsHEIGHT/2.6));
	~currentrandomseedstrext.background=~backgroundcolor;
	~currentrandomseedstrext.string ="Current random seeds:\n"++
	"randseed 1 - "++~randseed_1++"\n"++
	"randseed 2 - "++~randseed_2++"\n"++
	"randseed 3 - "++~randseed_3.asString++(if(~quicknewpiecebutton_num != 1, {
		"\n(also posted in post window)"++"\n"++
		"*make sure to hit "++"generate piece".quote++" to generate a new piece based on the new seeds"}, {""}));
	/*currentrandomseedstrext.align = \topLeft;*/
	~currentrandomseedstrext.stringColor=~stringcolor;
	~currentrandomseedstrext.font = Font(~default_font, size: ~firstviewboundsWIDTH/58.384615384615);

	~see_saved_seeds_file_button = Button.new(~firstview, Rect(~firstviewboundsWIDTH/1.4173228346457, ~firstviewboundsHEIGHT/1.45, ~firstviewboundsWIDTH/3.7, ~firstviewboundsHEIGHT/21)).states_([["See Saved Random Seeds",~stringcolor,~backgroundcolor],["See Saved Random Seeds",~stringcolor,~backgroundcolor]]).action_({arg button;

		h = Document.open((~ddbot_root)++"/program_files/other_program_files/duranduranbot_saved_random_seeds.scd");


	});
	~see_saved_seeds_file_button.font = Font(~default_font, size: ~firstviewboundsWIDTH/48);

	~training_settings_page_button = Button.new(~firstview, Rect(~firstviewboundsWIDTH/1.4173228346457, ~firstviewboundsHEIGHT/1.33, ~firstviewboundsWIDTH/3.7, ~firstviewboundsHEIGHT/21)).states_([["Settings",~stringcolor,~backgroundcolor]]).action_({arg button;
		~player_settings_page_func.value;
	});
	~training_settings_page_button.font = Font(~default_font, size: ~firstviewboundsWIDTH/48);

	~playerchoicetext = StaticText(~firstview, Rect(~firstviewboundsWIDTH/3, ~firstviewboundsHEIGHT/12, ~firstviewboundsWIDTH/2.8, ~firstviewboundsHEIGHT/21));
	~playerchoicetext.background=~backgroundcolor;
	~playerchoicetext.stringColor=~stringcolor;
	~playerchoicetext.string = "Settings - "++(
		case
		{FileReader.read((~ddbot_root)++"/program_files/other_program_files/which_players.scd", false, true).flatten.at(0) == 0.asString}{"All Players"}
		{FileReader.read((~ddbot_root)++"/program_files/other_program_files/which_players.scd", false, true).flatten.at(0) == 1.asString}{"Player 1"}
		{FileReader.read((~ddbot_root)++"/program_files/other_program_files/which_players.scd", false, true).flatten.at(0) == 2.asString}{"Player 2"}
		{FileReader.read((~ddbot_root)++"/program_files/other_program_files/which_players.scd", false, true).flatten.at(0) == 3.asString}{"Player 3"};
	)++", "++(
		case
		{FileReader.read((~ddbot_root)++"/program_files/other_program_files/numberofvoices.scd", false, true).flatten.at(0) == 1.asString}{"1 voice"}
		{FileReader.read((~ddbot_root)++"/program_files/other_program_files/numberofvoices.scd", false, true).flatten.at(0) == 2.asString}{"2 voices"}
		{FileReader.read((~ddbot_root)++"/program_files/other_program_files/numberofvoices.scd", false, true).flatten.at(0) == 3.asString}{"3 voices"};
	)++", A4 "++(~tuning_frequency)++" hz";
	~playerchoicetext.align = \topLeft;
	~playerchoicetext.font = Font(~default_font, size: ~firstviewboundsWIDTH/55);

	~save_current_piece_button = Button.new(~firstview, Rect(~firstviewboundsWIDTH/1.4173228346457, ~firstviewboundsHEIGHT/1.23, ~firstviewboundsWIDTH/3.7, ~firstviewboundsHEIGHT/21)).states_([["Save Current Piece",~stringcolor,~backgroundcolor]]).action_({arg button;

		"please wait".postln;
		g = File((~ddbot_root)++"/program_files/saved_scores/score_"++(z=((Date.getDate.format("%d_%m_%Y__%H_%M_%S").asString)))++".scd", "w");
		g.write("//");
		g.close;
		~re = Routine({
			~number_of_instruments.do(x = 0; {
				g = File((~ddbot_root)++"/program_files/saved_scores/score_"++(z)++".scd", "a");
				g.write("\n~instrument_"++(x=x+1)++"_full_midi = [ ");
				g.close;
				0.25.wait;
				y = (-1);
				g = File((~ddbot_root)++"/program_files/saved_scores/score_"++(z)++".scd", "a");
				(("instrument_"++(x)++"_full_midi").asSymbol.envirGet.size-1).do({
					g.write((("instrument_"++(x)++"_full_midi").asSymbol.envirGet.at(y=y+1)).asString++", ");

				});
				g.close;
				0.25.wait;
				g = File((~ddbot_root)++"/program_files/saved_scores/score_"++(z)++".scd", "a");
				g.write((("instrument_"++(x)++"_full_midi").asSymbol.envirGet.at(y=y+1)).asString++" ];");
				g.close;
			});
			"score saved".postln;
			("see scores in the "++"program files".quote++" folder").postln;
			~re.stop;
		});
		~re.play;

	});
	~save_current_piece_button.font = Font(~default_font, size: ~firstviewboundsWIDTH/48);

	~closeddbotbutton = Button.new(~firstview, Rect(~firstviewboundsWIDTH/1.2, ~firstviewboundsHEIGHT/1.1, ~firstviewboundsWIDTH/7.5, ~firstviewboundsHEIGHT/21)).states_([["Close DD'bot",~stringcolor,~backgroundcolor],["Close DD'bot",~stringcolor,~backgroundcolor]]).action_({arg button;
		if(~duranduranbot_routine.isPlaying == true, {~stop_func.value;});
		Server.freeAll;
		w.close;
	});
	~closeddbotbutton.font = Font(~default_font, size: ~firstviewboundsWIDTH/48);


	if(~gen_4_assit == nil, {~generatepiecebutton.valueAction = 0; ~gen_4_assit = 1;});

};