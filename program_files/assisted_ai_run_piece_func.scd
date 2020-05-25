// RUN/PLAY PIECE

~assisted_ai_run_piece_func = {

	if(~gen_new == 1, {~gen_new = 0; ~gen_new1 = 1;});

	if(~test_trim != 1, {
		~size_val = (("assist_inst_"++(1)++"_array_part_"++(~assist_part_number)).asSymbol.envirGet.size-1);
	}, {
		~size_val = ~cut_to - 1;
	});

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
						\freq, (~freq_r = ("assist_inst_"++(~x)++"_array_part_"++(~assist_part_number)).asSymbol.envirGet.at(~piece_position)),
						\amp, ~fr_func.value
					]);

				);});
			});
		};

		~init_func = {

			Server.default.makeBundle(Server.default.latency, {

				~number_of_instruments.do(~x = 0; {("duranduranbotsynth"++(~x=~x+1)).asSymbol.envirGet.set(
					\freq, (~freq_r = ("assist_inst_"++(~x)++"_array_part_"++(~assist_part_number)).asSymbol.envirGet.at(~piece_position)),
					\amp, ~fr_func.value

				);});
			});
		};

		if(~set_trim_on != 1, {

			~which_timeslider_func = {

				~timeslider.value = ~piece_position;
			};
		}, {
			if(~test_trim != 1, {
				~which_timeslider_func = {

					~cutslider1.value = ~piece_position;
				};
			}, {
				case
				{~cutslider_1_position > ~cutslider_2_position}{
					~first_cut_slider = 2;
					~which_timeslider_func = {

						~cutslider2.value = ~piece_position;
					};
				}
				{~cutslider_1_position < ~cutslider_2_position}{
					~first_cut_slider = 1;
					~which_timeslider_func = {

						~cutslider1.value = ~piece_position;
					};
				}
				{~cutslider_1_position == ~cutslider_2_position}{
					~first_cut_slider = 1;
					~which_timeslider_func = {

						~cutslider1.value = ~piece_position;
					};
				};
			});

		});

		~bit_timer_func = {

			~which_timeslider_func.value;


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

			/*if(~loop_on == 1, {~piece_position = ~pos_1}, {~piece_position = 0});*/

			if(~test_trim != 1, {~piece_position = 0;}, {~piece_position = ~cut_from - 1; ~test_trim = 0; ~test_trim2 = 1;});

			1.do({



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

				~bit_duration.wait;

				if((~piece_position >= ~size_val) /*or: (numberofloops >= ~num_loops)*/, {
					/*var amp = ~amp;
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
					1.wait; */
					AppClock.sched(0, {
						~stop_func.value;

						if(~gen_new1 == 1, {
							~gen_new1 = 0;

							if(("gen_rout_part_"++(~gen_assist_part_number)).asSymbol.envirGet.isPlaying == true, {
								("gen_rout_part_"++(~gen_assist_part_number)).asSymbol.envirGet.stop;
							});

							("gen_rout_part_"++(~gen_assist_part_number)).asSymbol.envirPut(

								Routine({

									1.do({

										30.wait;

										if(~assist_ml_on == 1, {

											~number_of_instruments.do(s = 0; {
												s=s+1;
												("instrument_"++(s)++"_full_midi_ml").asSymbol.envirPut(
													("assist_inst_"++(s)++"_array_part_"++(~gen_assist_part_number)++"_midi").asSymbol.envirGet;
												);
												("instrument_"++(s)++"_comp_interval_array_ml").asSymbol.envirPut(
													("assist_inst_"++(s)++"_comp_interval_array_part_"++(~gen_assist_part_number)).asSymbol.envirGet;
												);
												("instrument_"++(s)++"_comp_choice_array_ml").asSymbol.envirPut(
													("assist_inst_"++(s)++"_comp_choice_array_part_"++(~gen_assist_part_number)).asSymbol.envirGet;
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
						});

						if(~test_trim2 == 1, {
							~test_trim2 = 0;
							case
							{~first_cut_slider == 2}{~cutslider2.value = ~cut_from_old - 1;}
							{~first_cut_slider == 1}{~cutslider1.value = ~cut_from_old - 1;};
						});
					});
					/*if(~will_record == 1, {Server.default.stopRecording;});*/
					/*if((FileReader.read((~ddbot_root)++"/program_files/other_program_files/autonewpiece.scd", false, true).flatten.at(0) == 1.asString), {AppClock.sched(0, {~quicknewpiecebutton.valueAction = 0;});});
					*/
					~duranduranbot_routine.stop;
				});

			});

		});


		Server.default.makeBundle(Server.default.latency, {
			/*	~timer_routine.start;*/
			~duranduranbot_routine.play;
		});

	);
};