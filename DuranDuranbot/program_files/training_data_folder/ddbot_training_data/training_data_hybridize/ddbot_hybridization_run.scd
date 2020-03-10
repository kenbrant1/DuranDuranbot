(

// HOW TO RUN -
// click anywhere on page, and hit "cmd/ctrl - enter" to run ddbot training data hybridization.

// result will be "training_data_current" located in the "ddbot_training_data_hybridize_result" folder.








// How to hybridize DuranDuranbot training data, and what is it....
//
//
//
// ... What is DDbot training data hybridization?
//
//
// That is to hybridize the training data of two different training data sets.
// So for instance, to take two seperate trained DDbot programs, and to
// hybridize their data... kind of like animal mating.
//
//
//
// ... How do I hybridize DDBot training data?
//
//
// Open the "ddbot_training_data_hybridize" folder in which this help file is
// found. Then open both the "training_data_set_1" and "training_data_set_2" folders, and place DDbot training data in those folders, in the exact same folder format as the "training_data_current" folder in the "ddbot_training_data_folder" folder. For example - In the "training_data_set_1" folder, you would have a folder titled "training_data_current", with the exact same folder content as "ddbot_training_data_folder/training_data_current".... Just with different training data.
//
// The "training_data_set_1" and "training_data_set_1" folders both follow that format by defualt when you download this program.
//
//
//
// Then once that is done, open the "ddbot_hybridization_run.scd" file in the "ddbot_training_data_hybridize" folder... and run it follow the intructions on the top of the file.
//
//
// Then the resulting hybridized DDbot training data will be located in the "ddbot_training_data_hybridize_result" folder. To use that hybridized training data in the DDbot program, copy the "training_data_current" folder located in the "ddbot_training_data_hybridize_result" folder, and overwrite the "training_data_current" folder located in the "ddbot_training_data_folder" folder by pasting it in that location.
//
//
//
// ... If you get an error of some kind while running the hybridization.... Make sure that the correct folder/file formats are being used.



if(~number_of_instruments == nil, {~number_of_instruments = 3;});
~interval_array = (1..12);
~time_array = [0.0625, 0.125, 0.25, 0.5, 1, 2, 4, 8];
~dna_choice_array = [1, 2, 4, 8, 16, 32, 64, 128];




//hybridize "interval_weights"

~number_of_instruments.do(w = 0; v = 0; {
	w=w+1;
	(~interval_array.size+1).do(y = (-1); {
		y=y+1;
		(~interval_array.size+1).do(z = (-1); {
			z=z+1;
			/*("player_"++(w)++"__"++(y)++"_"++(z=z+1)++"_weights").asSymbol.envirPut(Array.fill(~interval_array.size, {1/~interval_array.size}).normalizeSum;);
			*//*g = File(Document.current.dir++"/other_program_files/ddbot_training_data_folder/training_data_current/interval_weights/player_"++(w)++"__"++(y)++"_"++(z)++"_weights.scd", "w");*/
			h = FileReader.read(Document.current.dir++"/training_data_set_1/training_data_current/interval_weights/player_"++(w)++"/"++(y)++"_"++(z)++"_weights.scd", false, true);
			h = h.flatten.asFloat;
			h = h.normalizeSum;
			k = FileReader.read(Document.current.dir++"/training_data_set_2/training_data_current/interval_weights/player_"++(w)++"/"++(y)++"_"++(z)++"_weights.scd", false, true);
			k = k.flatten.asFloat;
			k = k.normalizeSum;
			~interval_array.size.do(s = (-1); ~hybridize_array = []; {
				s=s+1;
				~hybridize_array = ~hybridize_array.add((h.at(s) + k.at(s)).value);
			});
			~hybridize_array = ~hybridize_array.normalizeSum;
			g = File(Document.current.dir++"/ddbot_training_data_hybridize_result/training_data_current/interval_weights/player_"++(w)++"/"++(y)++"_"++(z)++"_weights.scd", "w");
			~interval_array.size.do(s = (-1); {
				g.write(
					(
						s=s+1;
						~hybridize_array.at(s);
						/*a = h.at(s) + k.at(s);
						if(a.isNegative == true, {a = a.neg});
						a = a/2;
						case
						{h.at(s) > k.at(s)}{a = k.at(s) + a}
						{h.at(s) < k.at(s)}{a = h.at(s) + a}
						{h.at(s) == k.at(s)}{a = h.at(s)};
						a.value;*/

			).asString++" ");});
			g.close;
			v=v+1; v.postln;
		});
	});
});



//hybridize "time_weights"

~number_of_instruments.do(w = 0; v = 0; {
	w=w+1;
	(~time_array.size+1).do(x = (-1); {
		x=x+1;
		(~time_array.size+1).do(y = (-1); {
			y=y+1;
			(~time_array.size+1).do(z = (-1); {
				z=z+1;
				/*("player_"++(w)++"__"++(x)++"_"++(y)++"_"++(z=z+1)++"_time_weights").asSymbol.envirPut(Array.fill(~dna_choice_array.size, {1/~dna_choice_array.size}).normalizeSum;);
				*/
				h = FileReader.read(Document.current.dir++"/training_data_set_1/training_data_current/time_weights/player_"++(w)++"/"++(x)++"_"++(y)++"_"++(z)++"_time_weights.scd", false, true);
				h = h.flatten.asFloat;
				h = h.normalizeSum;
				k = FileReader.read(Document.current.dir++"/training_data_set_2/training_data_current/time_weights/player_"++(w)++"/"++(x)++"_"++(y)++"_"++(z)++"_time_weights.scd", false, true);
				k = k.flatten.asFloat;
				k = k.normalizeSum;
				~dna_choice_array.size.do(s = (-1); ~hybridize_array = []; {
					s=s+1;
					~hybridize_array = ~hybridize_array.add((h.at(s) + k.at(s)).value);
				});
				~hybridize_array = ~hybridize_array.normalizeSum;
				g = File(Document.current.dir++"/ddbot_training_data_hybridize_result/training_data_current/time_weights/player_"++(w)++"/"++(x)++"_"++(y)++"_"++(z)++"_time_weights.scd", "w");
				~dna_choice_array.size.do(s = (-1); {
					g.write(
						(
							s=s+1;
							~hybridize_array.at(s);
							/*a = h.at(s) + k.at(s);
							if(a.isNegative == true, {a = a.neg});
							a = a/2;
							case
							{h.at(s) > k.at(s)}{a = k.at(s) + a}
							{h.at(s) < k.at(s)}{a = h.at(s) + a}
							{h.at(s) == k.at(s)}{a = h.at(s)};
							a.value;*/

				).asString++" ");});
				g.close;
				v=v+1; v.postln;
			});
		});
	});
});



"\nddbot training data\nhybridization finished".postln;
);