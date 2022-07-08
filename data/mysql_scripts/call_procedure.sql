#CALL InsertRand(10000, 5, 0);
#CALL InsertOnes(10000);
use SakoDB;
select * from bayesian_features;
update bayesian_features set a=1 where id in (1);