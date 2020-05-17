// RUN/PLAY PIECE

~run_piece_func = {

	~size_val = ~instrument_1_full.size-1;

	~synthdefs_func.value;
	/*
	~low_pass_filter_on = 1;
	~gsine.value;*/

	(

		/*~init_start_func = {

		Server.default.makeBundle(Server.default.latency, {

		~number_of_instruments.do(~x = 0; {("f"++(~x=~x+1)).asSymbol.envirPut(("instrument_"++(~x)++"_full").asSymbol.envirGet.at(~piece_position));});
		~synthopen.value;
		});
		};

		~init_func = {

		Server.default.makeBundle(Server.default.latency, {

		~number_of_instruments.do(~x = 0; {("f"++(~x=~x+1)).asSymbol.envirPut(("instrument_"++(~x)++"_full").asSymbol.envirGet.at(~piece_position));});
		~synthflow.value;
		});

		};*/

		~fr_func = {(~amp - (~freq_r.cpsmidi-45*(0.7/36)))};

		~init_start_func = {

			Server.default.makeBundle(Server.default.latency, {

				~number_of_instruments.do(~x = 0; {("duranduranbotsynth"++(~x=~x+1)).asSymbol.envirPut(
					Synth("synth"++(~x), [
						\freq, (~freq_r = ("instrument_"++(~x)++"_full").asSymbol.envirGet.at(~piece_position)),
						\amp, ~fr_func.value
					]);

				);});
			});
		};

		~init_func = {

			Server.default.makeBundle(Server.default.latency, {

				~number_of_instruments.do(~x = 0; {("duranduranbotsynth"++(~x=~x+1)).asSymbol.envirGet.set(
					\freq, (~freq_r = ("instrument_"++(~x)++"_full").asSymbol.envirGet.at(~piece_position));,
					\amp, ~fr_func.value

				);});
			});
		};


		if(~ai_assist_compose == 0, {

			~bit_timer_func1 = {

				~rate_selection_end = ~piece_position;
				~rate_selection_start = ~piece_position-(~rate_time_range/~bit_duration);
				if(~rate_selection_start.isNegative == true, {~rate_selection_start = 0;});

					~piecepositionstatictext.string = (
						"Bit pos. "++(~piece_position.asInteger.asString)++" Midi "++(
							case
							{~number_of_instruments == 1}{(~instrument_1_full_midi.at(~piece_position).asString)}
							{~number_of_instruments == 2}{(~instrument_1_full_midi.at(~piece_position).asString)++" "++(~instrument_2_full_midi.at(~piece_position).asString)}
							{~number_of_instruments == 3}{(~instrument_1_full_midi.at(~piece_position).asString)++" "++(~instrument_2_full_midi.at(~piece_position).asString)++" "++(~instrument_3_full_midi.at(~piece_position).asString)};
						)
					);
			};
		}, {
			("assist_comp_part_"++(~assist_part_number)++"_button").asSymbol.envirGet.value = 0;
			("assist_comp_part_"++(~green = ~full_piece_part_array.at(0);)++"_button").asSymbol.envirGet.value = 1;

			~bit_timer_func1 = {
				if(~green != ~full_piece_part_array.at(~piece_position), {
					("assist_comp_part_"++(~green)++"_button").asSymbol.envirGet.value = 0;
					("assist_comp_part_"++(~green = ~full_piece_part_array.at(~piece_position))++"_button").asSymbol.envirGet.value = 1;
				});
			};
		});


		~bit_timer_func = {

			~bit_timer_func1.value;
			~timeslider.valueAction = ~piece_position;


			// old timer function

			/*bit_timer = bit_timer + 1;
			if(bit_timer >= ~bit_dur_rec, {*/
			/*AppClock.sched(0, {*/
			/*timeslider.valueAction = timeslider.value+1;*/
			/*~dd_secs = ~dd_secs-1;*/
			/*if(~dd_secs == (-1), {~dd_mins = ~dd_mins-1; ~dd_secs = 59;});*/
			/*piecedurationstatictext.string = ("Piece Length - "++(~piecedur_secs_zero.asInteger.asString)++" secs ("++(~dd_mins.asInteger.asString)++":"++(if(~dd_secs<10,{"0"},{""})++~dd_secs.asInteger.asString)++" mins)");*/
			/*if(timeslider.value >= ~rate_time_range, {rangeslider.valueAction = [timeslider.value-~rate_time_range, timeslider.value]}, {rangeslider.valueAction = [0, timeslider.value]});
			*//*if(~piecedur_secs_zero >= (~piecedur_secs), {~stop_func.value; "piece finished".postln; piecedurationstatictext.string = if(~fullpiecedur != nil, {~fullpiecedur}, {"piece length - --:-- mins"});})
			*//*if(~piece_position == ~instrument_1_full.size, {
			AppClock.sched(~bit_duration*4, {
			~stop_func.value; "piece finished".postln; piecedurationstatictext.string = if(~fullpiecedur != nil, {~fullpiecedur}, {"Piece Length - --:-- mins"});
			});
			});*/
			/*});
			bit_timer = 0;*/
			/*});*/
		};


		~duranduranbot_routine = Task({var bit_timer, numberofloops = 0;

			if(~loop_on == 1, {~piece_position = ~pos_1}, {~piece_position = 0});

			1.do({

				if(~will_record == 1, {
					thisProcess.platform.recordingsDir = ((~ddbot_root)++"/program_files/recordings");
					Server.default.record;
					1.wait;
				});

				"piece currently running".postln;

				0.5.wait;

				~init_start_func.value;

				AppClock.sched(0, {~bit_timer_func.value;});

				/*if(~will_record != 1, {~bit_timer_func.value;});*/

				~bit_duration.wait;

			});

			inf.do({

				~piece_position = ~piece_position + 1;

				~init_func.value;

				AppClock.sched(0, {~bit_timer_func.value;});

				/*if(~will_record != 1, {~bit_timer_func.value;});*/

				if(~loop_on == 1, {if(~piece_position == ~pos_2, {~piece_position = ~pos_1; numberofloops = numberofloops + 1; AppClock.sched(0, {~timeslider.valueAction = ~pos_1;});});});

				~bit_duration.wait;

				if((~piece_position >= (~size_val)) or: (numberofloops >= ~num_loops), {
					var amp = ~amp;
					150.do({~number_of_instruments.do({

						~amp = ~amp - (~amp/40);
						Server.default.makeBundle(Server.default.latency, {
							~number_of_instruments.do(j = 0; {("duranduranbotsynth"++(j=j+1)).asSymbol.envirGet.set(
								\amp, ~amp
							);});
						});
						0.005.wait;});});
					Server.default.makeBundle(Server.default.latency, {
						~number_of_instruments.do(j = 0; {("duranduranbotsynth"++(j=j+1)).asSymbol.envirGet.set(
							\amp, 0
						);});
					});
					~amp = 1;
					AppClock.sched(0, {~stop_func.value;});
					1.wait; if(~will_record == 1, {Server.default.stopRecording;});
					"piece finished".postln;
					if((FileReader.read((~ddbot_root)++"/program_files/other_program_files/autonewpiece.scd", false, true).flatten.at(0) == 1.asString), {AppClock.sched(0, {~quicknewpiecebutton.valueAction = 0;});});
				});

			});

		});


		Server.default.makeBundle(Server.default.latency, {
			/*	~timer_routine.start;*/
			~duranduranbot_routine.play;
		});

	);
};