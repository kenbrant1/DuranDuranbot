~main_page_func = {

	var arrow, newrandomseedsbutton, reversenewrandomseedsbutton, currentrandomseedstrext, playnil, see_saved_seeds_file_button;

	~firstview.close;

	~firstview = View(w, Rect(0, 0, w.bounds.width, w.bounds.height));
	/*view.clearOnRefresh = false;*/
	~firstview.background = ~backgroundcolor;
	~firstviewboundsWIDTH = ~firstview.bounds.width;
	~firstviewboundsHEIGHT = ~firstview.bounds.height;

	~firstviewtext1 = StaticText(~firstview, Rect(~firstviewboundsWIDTH/2.2857142857143, ~firstviewboundsHEIGHT/43.7, ~firstviewboundsWIDTH/8, ~firstviewboundsHEIGHT/21));
	~firstviewtext1.background=~backgroundcolor;
	~firstviewtext1.stringColor_(~stringcolor);
	~firstviewtext1.string = "Main Page";
	~firstviewtext1.align = \topLeft;
	~firstviewtext1.font = Font(size: ~firstviewboundsWIDTH/48);

	~firstviewtext2 = Button.new(~firstview, Rect(~firstviewboundsWIDTH/72, ~firstviewboundsHEIGHT/43.7, ~firstviewboundsWIDTH/16, ~firstviewboundsHEIGHT/21)).states_([["Back",~stringcolor,~backgroundcolor],["Back",~stringcolor,~backgroundcolor]]).action_({arg button;
		~firstview.close;
		~front_page_func.value;
	});

	~firstviewtext2.string = "Back";
	~firstviewtext2.font = Font(size: ~firstviewboundsWIDTH/48);

	~firstviewtext3 = Button.new(~firstview, Rect(~firstviewboundsWIDTH/1.3584905660377, ~firstviewboundsHEIGHT/43.7, ~firstviewboundsWIDTH/3.8, ~firstviewboundsHEIGHT/21)).states_([["Click Here For Instructions",~stringcolor,~backgroundcolor],["Click Here For Instructions",~stringcolor,~backgroundcolor]]).action_({arg button;
		~instructions_page_func.value;

	});
	~firstviewtext3.font = Font(size: ~firstviewboundsWIDTH/48);

	~generatepiecebutton = Button.new(~firstview, Rect(~firstviewboundsWIDTH/72, ~firstviewboundsHEIGHT/6.2428571428571, ~firstviewboundsWIDTH/6.5454545454545, ~firstviewboundsHEIGHT/21)).states_([["Generate Piece",~stringcolor,~backgroundcolor],["Generate Piece",~stringcolor,~backgroundcolor]]).action_({arg button;
		var piecegenerated;
		~piecegenerated = StaticText(~firstview, Rect(~firstviewboundsWIDTH/72, ~firstviewboundsHEIGHT/9.5, ~firstviewboundsWIDTH/6.2608695652174, ~firstviewboundsWIDTH/36));
		~piecegenerated.background=~backgroundcolor;
		~piecegenerated.stringColor_(~stringcolor);
		"Please Wait".postln;
		~piecegenerated.font = Font(size: ~firstviewboundsWIDTH/55);
		~piecegenerated.string = "Please Wait";/*
		if(~piece_created == "true", {~playbutton.close; ~arrow.close;});*/
		~piece_generator_func.value;

	});
	~generatepiecebutton.font = Font(size: ~firstviewboundsWIDTH/48);

	/*if(playnil == nil, {nil}, {*/
	/*~play_visible_func = {
	~play_visible_func_verify*/

	~playbutton_func = {
		~playbutton = Button.new(~firstview, Rect(~firstviewboundsWIDTH/4.95, ~firstviewboundsHEIGHT/6.2428571428571, ~firstviewboundsWIDTH/8, ~firstviewboundsHEIGHT/21)).states_([["Play Piece",~stringcolor,~backgroundcolor],["Pause Piece",~stringcolor,~backgroundcolor]]).action_({arg button;
			var pause_piece;
			if(button.value == 1, {
				if(~pause_piece != 1, {
					~seedscheck = 1;
					~stopbuttonfunc.value;
					~run_piece_func.value;
				}, {/*~timer_routine.resume;*/
					~amp = 1;
					Server.default.makeBundle(Server.default.latency, {~number_of_instruments.do(j = 0; {("duranduranbotsynth"++(j=j+1)).asSymbol.envirGet.set(\amp, ~amp);});});
					~duranduranbot_routine.resume; "Piece Resumed".postln; if(~will_record == 1, {s.record;});});
				~pause_piece = 0;
			}, {
				/*~timer_routine.pause;*/
				Server.default.makeBundle(Server.default.latency, {~number_of_instruments.do(j = 0; {("duranduranbotsynth"++(j=j+1)).asSymbol.envirGet.set(\amp, 0);});});

				~duranduranbot_routine.pause;
				/*Server.default.makeBundle(Server.default.latency, {~number_of_instruments.do(x = 0; {("duranduranbotsynth"++(x=x+1)).asSymbol.envirGet.set(\freq, 0);});});
				*/"Piece Paused".postln;
				if(~will_record == 1, {s.pauseRecording;});
				~pause_piece = 1;
				~playbutton.string = "Resume";
			});

		});
		~playbutton.font = Font(size: ~firstviewboundsWIDTH/48);
		/*};*/
		~arrow = StaticText(~firstview, Rect(~firstviewboundsWIDTH/5.7016393442623, ~firstviewboundsHEIGHT/6.2428571428571, ~firstviewboundsWIDTH/40, ~firstviewboundsHEIGHT/21));
		~arrow.background=~backgroundcolor;
		~arrow.stringColor_(~stringcolor);
		~arrow.string = "---";
		~arrow.font = Font(size: ~firstviewboundsWIDTH/48);
	};

	~playbutton_func.value;
	if(~pause_piece == 1, {~playbutton.string = "Resume"; ~stopbuttonfunc.value;});

	if(~piece_created == nil, {~playbutton.close; ~arrow.close;});

	~stop_func = {
		AppClock.sched(0, {
			/*~timer_routine.stop;*/
			/*Server.default.makeBundle(Server.default.latency, {~number_of_instruments.do(j = 0; {("duranduranbotsynth"++(j=j+1)).asSymbol.envirGet.set(\amp, 0);});});
			*/	/*~piece_position = 0;*/
			~timeslider.valueAction = 0;
			"piece stopped".postln;
			~fullpiecedur2 = ~fullpiecedur;
			if(~stopbutton != nil, {~stopbutton.close});
			~playbutton.value = 0;
			~pause_piece = 0;
			~playbutton.string = "Play Piece";
			Server.default.makeBundle(Server.default.latency, {~number_of_instruments.do(x = 0; {("duranduranbotsynth"++(x=x+1)).asSymbol.envirGet.free;});});
			~duranduranbot_routine.stop;
			if(~will_record == 1, {s.stopRecording;});
			Server.freeAll;
		});
	};

	~stopbuttonfunc = {~stopbutton = Button.new(~firstview, Rect(~firstviewboundsWIDTH/2.93, ~firstviewboundsHEIGHT/6.2428571428571, ~firstviewboundsWIDTH/8, ~firstviewboundsHEIGHT/21)).states_([["Stop piece",~stringcolor,~backgroundcolor]]).action_({arg button;
		~stop_func.value;
	});

	~stopbutton.font = Font(size: ~firstviewboundsWIDTH/48);
	};

	if(~duranduranbot_routine.isPlaying == true, {~playbutton.value = 1; ~stopbuttonfunc.value}, {~playbutton.value = 0});

	~net_change_mult_array = [1, 2, 4, 8, 16];

	~rate_button_1 = Button.new(~firstview, Rect(~firstviewboundsWIDTH/5.29, ~firstviewboundsHEIGHT/3.228571428571, ~firstviewboundsWIDTH/12, ~firstviewboundsHEIGHT/21)).states_([["x1 pts",~stringcolor,~backgroundcolor], ["x1 pts",~stringcolor,Color.blue]]).action_({arg button;
		~net_change_mult = ~net_change_mult_array.at(0);
		~rate_button_1.value = 1;
		~rate_button_2.value = 0;
		~rate_button_3.value = 0;
		~rate_button_4.value = 0;
		~rate_button_5.value = 0;
	});
	~rate_button_1.font = Font(size: ~firstviewboundsWIDTH/48);

	~rate_button_2 = Button.new(~firstview, Rect(~firstviewboundsWIDTH/3.62, ~firstviewboundsHEIGHT/3.228571428571, ~firstviewboundsWIDTH/12, ~firstviewboundsHEIGHT/21)).states_([["x2 pts",~stringcolor,~backgroundcolor], ["x2 pts",~stringcolor,Color.blue]]).action_({arg button;
		~net_change_mult = ~net_change_mult_array.at(1);
		~rate_button_1.value = 0;
		~rate_button_2.value = 1;
		~rate_button_3.value = 0;
		~rate_button_4.value = 0;
		~rate_button_5.value = 0;
	});
	~rate_button_2.font = Font(size: ~firstviewboundsWIDTH/48);

	~rate_button_3 = Button.new(~firstview, Rect(~firstviewboundsWIDTH/2.75, ~firstviewboundsHEIGHT/3.228571428571, ~firstviewboundsWIDTH/12, ~firstviewboundsHEIGHT/21)).states_([["x4 pts",~stringcolor,~backgroundcolor], ["x4 pts",~stringcolor,Color.blue]]).action_({arg button;
		~net_change_mult = ~net_change_mult_array.at(2);
		~rate_button_1.value = 0;
		~rate_button_2.value = 0;
		~rate_button_3.value = 1;
		~rate_button_4.value = 0;
		~rate_button_5.value = 0;
	});
	~rate_button_3.font = Font(size: ~firstviewboundsWIDTH/48);

	~rate_button_4 = Button.new(~firstview, Rect(~firstviewboundsWIDTH/2.22, ~firstviewboundsHEIGHT/3.228571428571, ~firstviewboundsWIDTH/12, ~firstviewboundsHEIGHT/21)).states_([["x8 pts",~stringcolor,~backgroundcolor], ["x8 pts",~stringcolor,Color.blue]]).action_({arg button;
		~net_change_mult = ~net_change_mult_array.at(3);
		~rate_button_1.value = 0;
		~rate_button_2.value = 0;
		~rate_button_3.value = 0;
		~rate_button_4.value = 1;
		~rate_button_5.value = 0;
	});
	~rate_button_4.font = Font(size: ~firstviewboundsWIDTH/48);

	~rate_button_5 = Button.new(~firstview, Rect(~firstviewboundsWIDTH/1.855, ~firstviewboundsHEIGHT/3.228571428571, ~firstviewboundsWIDTH/12, ~firstviewboundsHEIGHT/21)).states_([["x16 pts",~stringcolor,~backgroundcolor], ["x16 pts",~stringcolor,Color.blue]]).action_({arg button;
		~net_change_mult = ~net_change_mult_array.at(4);
		~rate_button_1.value = 0;
		~rate_button_2.value = 0;
		~rate_button_3.value = 0;
		~rate_button_4.value = 0;
		~rate_button_5.value = 1;
	});
	~rate_button_5.font = Font(size: ~firstviewboundsWIDTH/48);

	~rate_button = Button.new(~firstview, Rect(~firstviewboundsWIDTH/72, ~firstviewboundsHEIGHT/3.228571428571, ~firstviewboundsWIDTH/12, ~firstviewboundsHEIGHT/21)).states_([["Rate",~backgroundcolor,Color.green]]).action_({arg button;

		if(~duranduranbot_routine.isPlaying == true, {
			q = FileReader.read((~ddbot_root)++"/program_files/other_program_files/ddbot_machine_learning_on_off.scd");
			2.do({q = q.at(0)});
			q = q.asInteger;
			if(q == 1, {~machine_learning_func.value;});
			("\nlast "++(~rate_time_range)++" secs rated:\nx"++(~net_change_mult)++" points\n").postln;
		}, {
			"pice must be running to rate".postln;
			if(~ratedonetext != nil, {~ratedonetext.close; ~ratedonetext = nil;});
			~ratedonetext = StaticText(~firstview, Rect(~firstviewboundsWIDTH/3.063829787234, ~firstviewboundsHEIGHT/4.228571428571, ~firstviewboundsWIDTH/3.3, ~firstviewboundsHEIGHT/21));
			~ratedonetext.background=~backgroundcolor;
			~ratedonetext.stringColor_(~stringcolor);
			~ratedonetext.string = "PIECE MUST BE RUNNING TO RATE";
			~ratedonetext.font = Font(size: ~firstviewboundsWIDTH/55);

			AppClock.sched(3, {if(~ratedonetext != nil, {~ratedonetext.close; ~ratedonetext = nil;});});
		});
	});
	~rate_button.font = Font(size: ~firstviewboundsWIDTH/48);

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
	rate_revert_button.font = Font(size: ~firstviewboundsWIDTH/68);*/

	~ratepiecetext = StaticText(~firstview, Rect(~firstviewboundsWIDTH/72, ~firstviewboundsHEIGHT/2.67571428571, ~firstviewboundsWIDTH/1.5, ~firstviewboundsHEIGHT/12));
	~ratepiecetext.background=~backgroundcolor;
	~ratepiecetext.stringColor_(~stringcolor);
	~ratepiecetext.string = "^rate last "++(~rate_time_range)++" seconds from current position in piece by clicking on a xpt rating above, and clicking "++"Rate".quote++". Higher point # = faster training, but less fine-tuning.";
	~ratepiecetext.font = Font(size: ~firstviewboundsWIDTH/55);

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
	rangeslider.font = Font(size: ~firstviewboundsWIDTH/55);*/

	/*rate_selectionstatictext = StaticText(~firstview, Rect(~firstviewboundsWIDTH/72, ~firstviewboundsHEIGHT/1.502222222222 /*~firstviewboundsHEIGHT/6.2428571428571*/, ~firstviewboundsWIDTH/2.7, ~firstviewboundsHEIGHT/21));
	rate_selectionstatictext.background=~stringcolor;
	rate_selectionstatictext.string = if(~fullpiecedur != nil, {~fullpiecedur}, {"piece length - --:-- mins"});
	rate_selectionstatictext.font = Font(size: ~firstviewboundsWIDTH/48);*/

	if(~piece_position == nil, {~piece_position = 0});

	~timeslider = EZSlider(~firstview,  bounds: (Rect(~firstviewboundsWIDTH/72, ~firstviewboundsHEIGHT/2.073, ~firstviewboundsWIDTH/1.5, ~firstviewboundsHEIGHT/21)), controlSpec: ControlSpec(0, ("instrument_"++(1)++"_full").asSymbol.envirGet.size, \lin, 1, 50, "bits"), action: {
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

	if(~pause_piece != nil, {if(~pause_piece != 1, {~timeslider.valueAction = 0;/*rangeslider.valueAction = [0, if(~piecedur_secs != nil, {~piecedur_secs}, {0});]*/nil});});

	/*if(~pause_piece == 0, {~piece_position = 0});*/

	~timeslidertext = StaticText(~firstview, Rect(~firstviewboundsWIDTH/72, ~firstviewboundsHEIGHT/1.84, ~firstviewboundsWIDTH/1.5, ~firstviewboundsHEIGHT/22.55));
	~timeslidertext.background=~backgroundcolor;
	~timeslidertext.stringColor=~stringcolor;
	~timeslidertext.string = "^Jump to location in current piece by using the slider just above";
	~timeslidertext.font = Font(size: ~firstviewboundsWIDTH/55);


	~quicknewpiecebutton = Button.new(~firstview, Rect(~firstviewboundsWIDTH/72, ~firstviewboundsHEIGHT/4.228571428571, ~firstviewboundsWIDTH/6.1454545454545, ~firstviewboundsHEIGHT/21)).states_([["Quick New Piece",~stringcolor,~backgroundcolor],["Quick New Piece",~stringcolor,~backgroundcolor]]).action_({arg button;
		var quicknewpiecewait;
		quicknewpiecewait = StaticText(~firstview, Rect(~firstviewboundsWIDTH/72, ~firstviewboundsHEIGHT/9.5, ~firstviewboundsWIDTH/6.2608695, ~firstviewboundsHEIGHT/22.55));
		quicknewpiecewait.background_(~backgroundcolor);
		quicknewpiecewait.stringColor_(~stringcolor);
		quicknewpiecewait.string = "please wait";
		quicknewpiecewait.font = Font(size: ~firstviewboundsWIDTH/55);
		"please wait".postln;
		if(~piecegenerated != nil, {~piecegenerated.close});
		~quicknewpiecebutton = 1;
		{if(~duranduranbot_routine.isPlaying == true, {~stop_func.value});}.value;
		AppClock.sched(0, {
			if(~seedscheck == 1, {~newrandomseedsbutton.valueAction = 0}, {~seedscheck = 1});
			~generatepiecebutton.valueAction = 0; //piece fires at end of piece gen func
		});
		quicknewpiecewait.close;
	});
	~quicknewpiecebutton.font = Font(size: ~firstviewboundsWIDTH/48);

	~quicknewpiecebutton_revert = Button.new(~firstview, Rect(~firstviewboundsWIDTH/5.4, ~firstviewboundsHEIGHT/4.228571428571, ~firstviewboundsWIDTH/7.9454545454545, ~firstviewboundsHEIGHT/21)).states_([["Revert Piece",~stringcolor,~backgroundcolor],["Revert Piece",~stringcolor,~backgroundcolor]]).action_({arg button;
		var quicknewpiecewait;
		quicknewpiecewait = StaticText(~firstview, Rect(~firstviewboundsWIDTH/72, ~firstviewboundsHEIGHT/9.5, ~firstviewboundsWIDTH/6.2608695, ~firstviewboundsHEIGHT/22.55));
		quicknewpiecewait.background_(~stringcolor);
		quicknewpiecewait.stringColor_(~stringcolor);
		quicknewpiecewait.string = "Please Wait";
		if(~piecegenerated != nil, {~piecegenerated.close});
		quicknewpiecewait.font = Font(size: ~firstviewboundsWIDTH/55);
		"please wait".postln;
		~quicknewpiecebutton = 1;

		if(~duranduranbot_routine.isPlaying == true, {~stop_func.value;});

		AppClock.sched(0, {
			if(~seedscheck == 1, {~reversenewrandomseedsbutton.valueAction = 0}, {~seedscheck = 1});
			~generatepiecebutton.valueAction = 0; //piece fires at end of piece gen func
		});
		quicknewpiecewait.close;
	});
	~quicknewpiecebutton_revert.font = Font(size: ~firstviewboundsWIDTH/48);

	~recordbutton = Button.new(~firstview, Rect(~firstviewboundsWIDTH/72, ~firstviewboundsHEIGHT/1.35, ~firstviewboundsWIDTH/9, ~firstviewboundsHEIGHT/21)).states_([["Record Off",~stringcolor,~backgroundcolor],["Record On",~backgroundcolor,Color.red]]).action_({arg button;
		if(button.value == 1, {
			~recordtext.string = "All pieces WILL be recorded";
			~will_record = 1;
		}, {~recordtext.string = "All pieces will NOT be recorded"; ~will_record = 0;});

	});
	~recordbutton.font = Font(size: ~firstviewboundsWIDTH/48);

	~recordtext = StaticText(~firstview, Rect(~firstviewboundsWIDTH/7.6, ~firstviewboundsHEIGHT/1.35, ~firstviewboundsWIDTH/2.2, ~firstviewboundsHEIGHT/21));
	~recordtext.background=~backgroundcolor;
	~recordtext.stringColor=~stringcolor;
	~recordtext.string = "All pieces will NOT be recorded";
	~recordtext.font = Font(size: ~firstviewboundsWIDTH/55);

	if(~will_record == 1, {~recordtext.string = "All pieces WILL be recorded"; ~recordbutton.value = 1;},
		{~recordtext.string = "All pieces will NOT be recorded"; ~recordbutton.value = 0;});

	~piecedurationstatictext = StaticText(~firstview, Rect(~firstviewboundsWIDTH/72, ~firstviewboundsHEIGHT/1.65 /*~firstviewboundsHEIGHT/6.2428571428571*/, ~firstviewboundsWIDTH/2.7, ~firstviewboundsHEIGHT/21));
	~piecedurationstatictext.background=~backgroundcolor;
	~piecedurationstatictext.stringColor=~stringcolor;
	~piecedurationstatictext.string = if(~fullpiecedur != nil, {~fullpiecedur}, {"Piece Length - --:-- mins"});
	~piecedurationstatictext.font = Font(size: ~firstviewboundsWIDTH/48);

	~piecepositionstatictext = StaticText(~firstview, Rect((~firstviewboundsWIDTH/72)+(~firstviewboundsWIDTH/2.95)+30, ~firstviewboundsHEIGHT/1.65 /*~firstviewboundsHEIGHT/6.2428571428571*/, ~firstviewboundsWIDTH/2.7, ~firstviewboundsHEIGHT/21));
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
	~piecepositionstatictext.font = Font(size: ~firstviewboundsWIDTH/48);

	if(~pos_1 == nil, {~pos_1 = 0});
	if(~pos_2 == nil, {~pos_2 = 0});

	~looptextfield_1 = TextField(~firstview, Rect(~firstviewboundsWIDTH/72, ~firstviewboundsHEIGHT/1.47, ~firstviewboundsWIDTH/10, ~firstviewboundsHEIGHT/21));
	~looptextfield_1.string = ~pos_1.asString;
	~looptextfield_1.background_(~backgroundcolor);
	~looptextfield_1.stringColor_(~stringcolor);
	~looptextfield_1.font = Font(size: ~firstviewboundsWIDTH/48);
	~looptextfield_1.action = {
		~sdfk = ~looptextfield_1.value.asInteger;
		/*	~pos_1 = looptextfield_1.value.asInteger;*/
		if(~sdfk.value.isInteger == true, {~pos_1 = ~sdfk.value.asInteger;}, {~looptextfield_1.string = "integer".asString;});
		~looptextfield_1.string = ~pos_1.asString;
	};

	~looptextfield_2 = TextField(~firstview, Rect(~firstviewboundsWIDTH/8, ~firstviewboundsHEIGHT/1.47, ~firstviewboundsWIDTH/10, ~firstviewboundsHEIGHT/21));
	~looptextfield_2.string = ~pos_2.asString;
	~looptextfield_2.background_(~backgroundcolor);
	~looptextfield_2.stringColor_(~stringcolor);
	~looptextfield_2.font = Font(size: ~firstviewboundsWIDTH/48);
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

	~loopsetbutton =  Button.new(~firstview, Rect(~firstviewboundsWIDTH/4.3, ~firstviewboundsHEIGHT/1.47, ~firstviewboundsWIDTH/8.5, ~firstviewboundsHEIGHT/21)).states_([["Loop is OFF",~stringcolor,~backgroundcolor],["Loop is ON",~stringcolor,~backgroundcolor]]).action_({arg button;

		if(button.value == 1, {~loop_on = 1;}, {~loop_on = 0;});
		/*if(~duranduranbot_routine.isPlaying == true, {~stop_func.value;});
		~loop_creator_func.value;*/
	});

	if(~loop_on == 1, {~loopsetbutton.value = 1}, {~loopsetbutton.value = 0});

	if(~num_loops == nil, {~num_loops = 24});

	~looptextfield_3 = TextField(~firstview, Rect(~firstviewboundsWIDTH/2.2, ~firstviewboundsHEIGHT/1.47, ~firstviewboundsWIDTH/10, ~firstviewboundsHEIGHT/21));
	~looptextfield_3.string = ~num_loops.asString;
	~looptextfield_3.background_(~backgroundcolor);
	~looptextfield_3.stringColor_(~stringcolor);
	~looptextfield_3.font = Font(size: ~firstviewboundsWIDTH/48);
	~looptextfield_3.action = {
		~sdfk = ~looptextfield_3.value.asInteger;
		if(~sdfk.value.isInteger == true, {~num_loops = ~sdfk.value.asInteger;}, {~looptextfield_3.string = "integer".asString;});
		~looptextfield_3.string = ~num_loops.asString;
	};

	/*ctrlperiodtext = StaticText(~firstview, Rect(~firstviewboundsWIDTH/1.4173228346457, ~firstviewboundsHEIGHT/13.5, ~firstviewboundsWIDTH/3.5, ~firstviewboundsHEIGHT/4.37));
	ctrlperiodtext.background=~backgroundcolor;
	ctrlperiodtext.string = "*Important Note*... You can kill all processes by hitting "++"cmd/ctrl - period".quote++"on your keyboard.";
	ctrlperiodtext.align = \topLeft;
	ctrlperiodtext.font = Font(size: ~firstviewboundsWIDTH/70.454545454545);*/

	~newrandomseedsbutton = Button.new(~firstview, Rect(~firstviewboundsWIDTH/1.4173228346457, ~firstviewboundsHEIGHT/6.2428571428571, ~firstviewboundsWIDTH/5.1428571428571, ~firstviewboundsHEIGHT/21)).states_([["New Random Seeds",~stringcolor,~backgroundcolor],["New Random Seeds",~stringcolor,~backgroundcolor]]).action_({arg button;
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
		"randseed 3 - "++~randseed_3.asString++(if(~quicknewpiecebutton != 1, {
			"\n(also posted in post window)"++"\n"++
			"*make sure to hit "++"generate piece".quote++" to generate a new piece based on the new seeds"}, {""}));
		z.postln;
		~currentrandomseedstrext.string = z;
		~currentrandomseedstrext.background=~backgroundcolor;
		~currentrandomseedstrext.stringColor=~stringcolor;

	});
	~newrandomseedsbutton.font = Font(size: ~firstviewboundsWIDTH/48);

	~reversenewrandomseedsbutton = Button.new(~firstview, Rect(~firstviewboundsWIDTH/1.1059907834101, ~firstviewboundsHEIGHT/6.2428571428571, ~firstviewboundsWIDTH/12, ~firstviewboundsHEIGHT/21)).states_([["Revert",~stringcolor,~backgroundcolor],["Revert",~stringcolor,~backgroundcolor]]).action_({arg button;
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
		"randseed 3 - "++~randseed_3.asString++(if(~quicknewpiecebutton != 1, {
			"\n(also posted in post window)"++"\n"++
			"*make sure to hit "++"generate piece".quote++" to generate a new piece based on the new seeds"}, {""}));
		z.postln;
		~currentrandomseedstrext.string = z;
		~currentrandomseedstrext.background=~backgroundcolor;
		~currentrandomseedstrext.stringColor=~stringcolor;
	});
	~reversenewrandomseedsbutton.font = Font(size: ~firstviewboundsWIDTH/48);

	~currentrandomseedstrext = TextView(~firstview, Rect(~firstviewboundsWIDTH/1.4173228346457, ~firstviewboundsHEIGHT/4.6, ~firstviewboundsWIDTH/3.56, ~firstviewboundsHEIGHT/2.6));
	~currentrandomseedstrext.background=~backgroundcolor;
	~currentrandomseedstrext.string ="Current random seeds:\n"++
	"randseed 1 - "++~randseed_1++"\n"++
	"randseed 2 - "++~randseed_2++"\n"++
	"randseed 3 - "++~randseed_3.asString++(if(~quicknewpiecebutton != 1, {
		"\n(also posted in post window)"++"\n"++
		"*make sure to hit "++"generate piece".quote++" to generate a new piece based on the new seeds"}, {""}));
	/*currentrandomseedstrext.align = \topLeft;*/
	~currentrandomseedstrext.stringColor=~stringcolor;
	~currentrandomseedstrext.font = Font(size: ~firstviewboundsWIDTH/58.384615384615);

	~see_saved_seeds_file_button = Button.new(~firstview, Rect(~firstviewboundsWIDTH/1.4173228346457, ~firstviewboundsHEIGHT/1.6, ~firstviewboundsWIDTH/4, ~firstviewboundsHEIGHT/21)).states_([["See Saved Random Seeds",~stringcolor,~backgroundcolor],["See Saved Random Seeds",~stringcolor,~backgroundcolor]]).action_({arg button;

		h = Document.open((~ddbot_root)++"/program_files/duranduranbot_saved_random_seeds.scd");


	});
	~see_saved_seeds_file_button.font = Font(size: ~firstviewboundsWIDTH/48);

	~newpieceautobutton = Button.new(~firstview, Rect(~firstviewboundsWIDTH/72, ~firstviewboundsHEIGHT/1.25, ~firstviewboundsWIDTH/5, ~firstviewboundsHEIGHT/21)).states_([["Auto New Piece On",~stringcolor,~backgroundcolor],["Auto New Piece Off",~stringcolor,~backgroundcolor]]).action_({arg button;
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
	~newpieceautobutton.font = Font(size: ~firstviewboundsWIDTH/48);

	~newpieceautotext = StaticText(~firstview, Rect(~firstviewboundsWIDTH/4.5, ~firstviewboundsHEIGHT/1.25, ~firstviewboundsWIDTH/2.2, ~firstviewboundsHEIGHT/21));
	~newpieceautotext.background=~backgroundcolor;
	~newpieceautotext.stringColor=~stringcolor;
	if((FileReader.read((~ddbot_root)++"/program_files/other_program_files/autonewpiece.scd", false, true).flatten.at(0) == 1.asString), {
		~newpieceautobutton.value = 0; ~newpieceautotext.string = "Automatic "++"Quick New Piece".quote++" is ON";
	}, {~newpieceautobutton.value = 1; ~newpieceautotext.string = "Automatic "++"Quick New Piece".quote++" is OFF";});
	~newpieceautotext.font = Font(size: ~firstviewboundsWIDTH/55);

	~machinelearningbutton = Button.new(~firstview, Rect(~firstviewboundsWIDTH/72, ~firstviewboundsHEIGHT/1.08, ~firstviewboundsWIDTH/5.5, ~firstviewboundsHEIGHT/21)).states_([["ML Training Is ON",~stringcolor,~backgroundcolor],["ML Training Is OFF",~stringcolor,~backgroundcolor]]).action_({arg button;
		if(button.value == 1, {
			g = File((~ddbot_root)++"/program_files/other_program_files/ddbot_machine_learning_on_off.scd", "w");
			g.write("0");
			g.close;
		}, {
			g = File((~ddbot_root)++"/program_files/other_program_files/ddbot_machine_learning_on_off.scd", "w");
			g.write("1");
			g.close;
		});

	});
	~machinelearningbutton.font = Font(size: ~firstviewboundsWIDTH/48);

	~machinelearningtext = StaticText(~firstview, Rect(~firstviewboundsWIDTH/5, ~firstviewboundsHEIGHT/1.08, ~firstviewboundsWIDTH/2, ~firstviewboundsHEIGHT/21));
	~machinelearningtext.background=~backgroundcolor;
	~machinelearningtext.stringColor=~stringcolor;
	~machinelearningtext.string = "More info about machine learning on "++"instructions".quote++" page";
	~machinelearningtext.font = Font(size: ~firstviewboundsWIDTH/55);

	q = FileReader.read((~ddbot_root)++"/program_files/other_program_files/ddbot_machine_learning_on_off.scd");
	2.do({q = q.at(0)});
	q = q.asInteger;
	if(q == 1, {~machinelearningbutton.value = 0;});

	~loadscorebuttonbutton = Button.new(~firstview, Rect(~firstviewboundsWIDTH/72, ~firstviewboundsHEIGHT/1.16, ~firstviewboundsWIDTH/5.8, ~firstviewboundsHEIGHT/21)).states_([["Load Saved Score",~stringcolor,~backgroundcolor]]).action_({arg button;

		if(~loaded_score.value == nil, {
			~loadscoretext.string = "Paste score in top of "++"duranduranbot_RUN.scd".quote++" file in "++"~loaded_score".quote++" function.";
			AppClock.sched(5, {~loadscoretext.string = "Load a saved score here. Then press "++"Play Piece".quote++".";});
		}, {
			~saved_piece_loaded = 1;
			~loaded_score.value;
			if(~duranduranbot_routine.isPlaying == true, {~stop_func.value;});
			~generatepiecebutton.valueAction = 1;
			~loadscoretext.string = "Score has been loaded. Press "++"Play Piece".quote++" to run.";
			AppClock.sched(5, {~loadscoretext.string = "Load a saved score here. Then press "++"Play Piece".quote++".";});
		});

	});
	~loadscorebuttonbutton.font = Font(size: ~firstviewboundsWIDTH/48);

	~loadscoretext = StaticText(~firstview, Rect(~firstviewboundsWIDTH/5.2, ~firstviewboundsHEIGHT/1.16, ~firstviewboundsWIDTH/1.3, ~firstviewboundsHEIGHT/21));
	~loadscoretext.background=~backgroundcolor;
	~loadscoretext.stringColor=~stringcolor;
	~loadscoretext.string = "Load a saved score here. Then press "++"Play Piece".quote++".";
	~loadscoretext.font = Font(size: ~firstviewboundsWIDTH/55);

	~training_settings_page_button = Button.new(~firstview, Rect(~firstviewboundsWIDTH/1.4173228346457, ~firstviewboundsHEIGHT/1.45, ~firstviewboundsWIDTH/4, ~firstviewboundsHEIGHT/21)).states_([["Settings",~stringcolor,~backgroundcolor]]).action_({arg button;
		~player_settings_page_func.value;
	});
	~training_settings_page_button.font = Font(size: ~firstviewboundsWIDTH/48);

	~playerchoicetext = StaticText(~firstview, Rect(~firstviewboundsWIDTH/3, ~firstviewboundsHEIGHT/12, ~firstviewboundsWIDTH/3, ~firstviewboundsHEIGHT/21));
	~playerchoicetext.background=~backgroundcolor;
	~playerchoicetext.stringColor=~stringcolor;
	~playerchoicetext.string = "Current Player(s) - "++(
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
	);
	~playerchoicetext.align = \topLeft;
	~playerchoicetext.font = Font(size: ~firstviewboundsWIDTH/55);

	~color_menu = PopUpMenu(~firstview, Rect(~firstviewboundsWIDTH/1.4173228346457, ~firstviewboundsHEIGHT/1.33, ~firstviewboundsWIDTH/4, ~firstviewboundsHEIGHT/21));
	~color_menu.items = [
		"GUI color - "++(
			case
			{FileReader.read((~ddbot_root)++"/program_files/other_program_files/gui_color.scd", false, true).flatten.at(0) == 0.asString}{"Black"}
			{FileReader.read((~ddbot_root)++"/program_files/other_program_files/gui_color.scd", false, true).flatten.at(0) == 1.asString}{"White"};
		),
		"Black", "White"];
	~color_menu.background_(~backgroundcolor);
	~color_menu.stringColor_(~stringcolor);
	~color_menu.value = FileReader.read((~ddbot_root)++"/program_files/other_program_files/gui_color.scd", false, true);
	~color_menu.action = { arg menu;
		g = File((~ddbot_root)++"/program_files/other_program_files/gui_color.scd", "w");
		g.write(
			case
			{~color_menu.value == 0}{nil}
			{~color_menu.value == 1}{"0".asString}
			{~color_menu.value == 2}{"1".asString};

		);
		g.close;
		case
		{~color_menu.value == 0}{nil}
		{~color_menu.value == 1}{
			~backgroundcolor = Color.black;
			~stringcolor = Color.white;
		}
		{~color_menu.value == 2}{
			~backgroundcolor = Color.white;
			~stringcolor = Color.black;
		};
		w.close;
		~duranduranbot_GUI_func.value;
		~main_page_func.value;
	};
	~color_menu.font = Font(size: ~firstviewboundsWIDTH/48);

	~save_current_piece_button = Button.new(~firstview, Rect(~firstviewboundsWIDTH/1.4173228346457, ~firstviewboundsHEIGHT/1.23, ~firstviewboundsWIDTH/4, ~firstviewboundsHEIGHT/21)).states_([["Save Current Piece",~stringcolor,~backgroundcolor]]).action_({arg button;

		"please wait".postln;
		g = File((~ddbot_root)++"/program_files/other_program_files/ddbot_saved_scores/score_"++(z=((Date.getDate.format("%d_%m_%Y__%H_%M_%S").asString)))++".scd", "w");
		g.write("// Copy and paste the all lines below in the "++"Load Saved Score".quote++" section in the top of the "++"duranduranbot_RUN.scd".quote++" file");
		g.close;
		~re = Routine({
			~number_of_instruments.do(x = 0; {
				g = File((~ddbot_root)++"/program_files/other_program_files/ddbot_saved_scores/score_"++(z)++".scd", "a");
				g.write("\n~instrument_"++(x=x+1)++"_full_midi = [ ");
				g.close;
				0.25.wait;
				y = (-1);
				g = File((~ddbot_root)++"/program_files/other_program_files/ddbot_saved_scores/score_"++(z)++".scd", "a");
				(("instrument_"++(x)++"_full_midi").asSymbol.envirGet.size-1).do({
					g.write((("instrument_"++(x)++"_full_midi").asSymbol.envirGet.at(y=y+1)).asString++", ");

				});
				g.close;
				0.25.wait;
				g = File((~ddbot_root)++"/program_files/other_program_files/ddbot_saved_scores/score_"++(z)++".scd", "a");
				g.write((("instrument_"++(x)++"_full_midi").asSymbol.envirGet.at(y=y+1)).asString++" ];");
				g.close;
			});
			"score saved".postln;
			("see scores in the "++"program files".quote++" folder").postln;
			~re.stop;
		});
		~re.play;

	});
	~save_current_piece_button.font = Font(size: ~firstviewboundsWIDTH/48);

	~closeddbotbutton = Button.new(~firstview, Rect(~firstviewboundsWIDTH/1.2, ~firstviewboundsHEIGHT/1.1, ~firstviewboundsWIDTH/7.5, ~firstviewboundsHEIGHT/21)).states_([["Close DD'bot",~stringcolor,~backgroundcolor],["Close DD'bot",~stringcolor,~backgroundcolor]]).action_({arg button;
		~stop_func.value;
		Server.freeAll;
		w.close;
	});
	~closeddbotbutton.font = Font(size: ~firstviewboundsWIDTH/48);

};