// SAVE RANDOM SEEDS TO FILE

~save_random_seeds_func = {

	a = Date.getDate;

	/*Document.open((~ddbot_root)++"/program_files/duranduranbot_saved_random_seeds.scd").close;*/

	h = File.new((~ddbot_root)++"/program_files/duranduranbot_saved_random_seeds.scd", "a+");

	h.write("\n\n"++a.asString++
		"\nrandom seed 1 - "++~randseed_1.asString++
		"\nrandom seed 2 - "++~randseed_2.asString++
		"\nrandom seed 3 - "++~randseed_3.asString
	);

	h.close;
};