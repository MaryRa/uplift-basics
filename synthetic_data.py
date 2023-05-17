import numpy as np
import pandas as pd


class UpliftDataSet:
    """Class to generate synthetic data for experiments with Uplift models.
    
Parameters
--------------------------------------------------------------------------------
age_min: int, the minimum value of age for users
age_max: int, the maximum value of age for users
n_cities: int, number of unique cities for users
n_users: int, number of users on the platform
share_negative_people: float, share people, who will react negatively if there 
                              will be communication with them 
--------------------------------------------------------------------------------
"""
    def __init__(self,
                 age_min: int = 16,
                 age_max: int = 60,
                 n_cities: int = 6,
                 n_users: int = 100000,
                 share_negative_people: float = 0.2):
        self._age_min = age_min
        self._age_max = age_max
        self._n_cities = n_cities
        self._n_users = n_users
        self._share_negative_people = share_negative_people
        
        self.rand_user_param_value = {}
        self._zero_communication_value = 0
        self.daily_prob_of_purchase = None
        
        self.user_info = self._get_user_data(self._n_users, self._age_min, 
                                             self._age_max, self._n_cities)
        self._negative_reaction_on_communication = self._get_negative_reaction(
            self._n_users, self._share_negative_people)
        
    
    def _get_user_data(self, n_users: int, age_min: int, 
                       age_max: int, n_cities: int) -> dict:
        """Generate users and their characteristics"""
        return {
            "id": np.arange(n_users),
            "gender": np.random.choice(['m', 'f'], size=n_users, p=[0.5, 0.5]),
             "age":  np.clip(
                 np.int_(
                     np.random.normal(
                         (age_min+age_max)/2, 
                         ((age_min+age_max)/2-age_min)/3,
                         size=n_users
                     )
                 ),
                 age_min, age_max),
             "city":  np.random.randint(0, n_cities, size=n_users)
        }
    
    @property
    def get_user_data(self) -> pd.DataFrame:
        """Return user info"""
        return pd.DataFrame(self.user_info)
    
    def _get_negative_reaction(self, n_users: int, share_negative_people: float
                              ) -> np.array:
        """Select those users who have negative reaction on communication"""
        return np.random.choice([True, False], 
                                p=[share_negative_people, 
                                   1-share_negative_people], 
                                size=n_users)
        
    def _get_uplift_params(self, user_info: dict, n_users: int) -> dict:
        """Select user characteristics which will define that users 
with this characteristics will react positively on communication """
        user_param = list(set(user_info.keys())-{'id'})
        n_rand_user_param = np.random.choice(len(user_param)) + 1
        rand_user_param = np.random.choice(user_param, 
                                           size=n_rand_user_param, 
                                           replace=False)
        rand_user_param_value = {}
        availible_users = ~self._negative_reaction_on_communication
        for x in rand_user_param:
            val = np.unique(user_info[x][availible_users])
            if x == 'age':
                min_max_age = np.random.choice(val, size=2, replace=False)
                rand_user_param_value[x] = np.arange(np.min(min_max_age), 
                                                     np.max(min_max_age)+1)
            else:
                rand_user_param_value[x] = np.random.choice(
                    val, size=np.random.choice(len(val) - 1) + 1, replace=False)
            availible_users = availible_users & np.isin(user_info[x],
                                                        rand_user_param_value[x]
                                                       )
        return {"vector_user_selection": availible_users, 
                "parameters": rand_user_param_value}
     
    def _get_prob_for_purchases(self, 
                                n_users: int, 
                                max_prob: float = 0.5
                               ) -> dict:
        """Return probability of purchases if there is no communication 
with users"""
        daily_prob_of_purchase = {self._zero_communication_value: 
                                  np.random.uniform(0, max_prob, size=n_users)}
        return daily_prob_of_purchase
    
    def _add_prob_for_purchase(self, 
                               daily_prob_of_purchase: dict, 
                               apply_for_users_bool: np.array,
                               communication_type: "int | str | tuple | float",
                               probability_increase: float,
                              ) -> dict:
        """Add changed probability distribution for communication_type"""
        daily_prob_of_purchase[communication_type] = daily_prob_of_purchase[
            self._zero_communication_value].copy()
        daily_prob_of_purchase[communication_type][
            apply_for_users_bool] *= probability_increase
        daily_prob_of_purchase[communication_type][
            apply_for_users_bool] = np.maximum(
            1, daily_prob_of_purchase[communication_type][apply_for_users_bool])
        daily_prob_of_purchase[communication_type][
            self._negative_reaction_on_communication] /= probability_increase
        daily_prob_of_purchase[communication_type][
            self._negative_reaction_on_communication] = np.minimum(
            0, daily_prob_of_purchase[communication_type][
                self._negative_reaction_on_communication])
        return daily_prob_of_purchase
    
    def _get_users_with_comm_and_without(self, 
                                         user_ids: np.array, 
                                         n_people_to_communicate: int,
                                         n_communication: int) -> list:
        """Devide user_ids on n_communication + 1 groups 
where the first n_communication groups are users with communication 
and the last one without communication"""
        user_id_com = np.random.choice(
            user_ids, size=n_people_to_communicate, replace=False)
        people_with_communication = np.array_split(
            user_id_com, n_communication)
        users_group = people_with_communication + [
            np.array(list(set(user_ids)-set(user_id_com)))]
        return users_group
    
    def _get_one_day_purchases(self, 
                               user_ids: dict, 
                               people_with_communication: np.array, 
                               daily_prob_of_purchase: dict, 
                               day: int) -> pd.DataFrame:
        """Return one day purchases for users according to daily probability of purchases"""
        communication = np.array([np.nan]*len(user_ids))
        prob_purch = daily_prob_of_purchase[self._zero_communication_value].copy()
        for i, key in enumerate(list(set(daily_prob_of_purchase.keys()) -
                                {self._zero_communication_value}) + 
                                [self._zero_communication_value]):
            
            ind = np.isin(user_ids, people_with_communication[i])
            communication[ind] = key
            prob_purch[ind] = daily_prob_of_purchase[key][ind]
        purchases = (np.random.uniform(0, 1, size=len(prob_purch)) <= prob_purch
                    )*1
        res = pd.DataFrame({"day": [day]*len(user_ids), 
                             "id": user_ids, 
                             "communication": communication, 
                             "purchases": purchases})
        return res[~res['communication'].isna()].reset_index(drop=True)
    
    def get_train(self, 
                  add_purchases_value: bool = True, 
                  check_median: float = 60, 
                  check_std: float = 10, 
                  sorted_type_of_communication: tuple = (1,),
                  communicate_the_same: bool = True,
                  subsample_for_train: float = 0.3,
                  share_communicate: float = 0.75,
                  probability_increase: float = 1.5,
                 n_days: int = 30,
                 ) -> pd.DataFrame:
        """Generate train for a model.
    
Parameters
--------------------------------------------------------------------------------
add_purchases_value: bool, do you want to add value of purchases? 
                           Otherwise it generates only quantity
check_median: float, median value of purchases
check_std: float, standard deviation value of purchases
sorted_type_of_communication: tuple, type of communications from weaker to 
                                     the strongest one. For example it can be 
                                     discounts (5, 10, 100)
communicate_the_same: bool, do you want to communicate to the same users every 
                            day? Otherwise new users for communication will 
                            be generated every day
subsample_for_train: float, share of users who go to train set (value must be 
                            in the interval (0,1])
share_communicate: float, share users from train who will receive communication
                          (value must be in the interval (0,1]))
probability_increase: float, how communication change probability to buy, 
                             it mus be > 1
n_days: int, number of days in dataset
--------------------------------------------------------------------------------

Returns
--------------------------------------------------------------------------------
pd.DataFrame(columns=[day, id, communication, purchases]) with orders
--------------------------------------------------------------------------------
"""
        self.rand_user_param_value = {}
        self._zero_communication_value = 0
        self.daily_prob_of_purchase = self._get_prob_for_purchases(self._n_users)
        
        for i, communication in enumerate(sorted_type_of_communication):
            self.rand_user_param_value[communication] = self._get_uplift_params(
                self.user_info, self._n_users)
            if i > 0:
                self.rand_user_param_value[communication][
                    'vector_user_selection'] = (self.rand_user_param_value[
                    communication]['vector_user_selection'] | 
                                                self.rand_user_param_value[
                    sorted_type_of_communication[i-1]]['vector_user_selection'])
            self.daily_prob_of_purchase = self._add_prob_for_purchase(
                self.daily_prob_of_purchase, 
                self.rand_user_param_value[communication]['vector_user_selection'],
                communication_type=communication,
                probability_increase=probability_increase)
            
        selected_user_ids = np.random.choice(
            self.user_info['id'], 
            size=int(len(self.user_info['id'])*subsample_for_train), 
            replace=False)
        n_people_to_communicate = int(int(len(self.user_info['id'])*
                                          subsample_for_train)
                                      * share_communicate)
                    
        if communicate_the_same:
            users_group = self._get_users_with_comm_and_without(
                user_ids=selected_user_ids, 
                n_people_to_communicate=n_people_to_communicate,
                n_communication=len(self.daily_prob_of_purchase.keys())-1)
            purchases = pd.concat([self._get_one_day_purchases(
                self.user_info['id'], 
                users_group,
                self.daily_prob_of_purchase,
                x) for x in range(n_days)])
        else:
            purchases = pd.concat([self._get_one_day_purchases(
                self.user_info['id'], 
                self._get_users_with_comm_and_without(
                    user_ids=selected_user_ids, 
                    n_people_to_communicate=n_people_to_communicate,
                    n_communication=len(self.daily_prob_of_purchase.keys())-1),
                self.daily_prob_of_purchase,
                x) for x in range(n_days)])
        if add_purchases_value:
            mu = np.log(check_median)
            sigma = np.log((1 + (1+4*1*check_std**2/np.exp(2*mu))**0.5)/2)**0.5
            purchases.loc[purchases['purchases'] == 1, 
                          'purchases'] = np.random.lognormal(
                mean=mu, sigma=sigma, size=np.sum(purchases['purchases']))
        purchases = pd.DataFrame(self.user_info).merge(purchases, on='id')
        return purchases
    
    def check_test(self, 
                   users_with_type_of_communication: pd.DataFrame, 
                   add_purchases_value: bool = True, 
                   check_median: float = 60, 
                   check_std: float = 10) -> pd.Series:
        """Check purchases for all users if we communicate with them
        
Parameters
--------------------------------------------------------------------------------
users_with_type_of_communication: pd.DataFrame, with required columns 
                                                id and communication
add_purchases_value: bool, do you want to add value of purchases? 
                           Otherwise it generates only quantity
check_median: float, median value of purchases
check_std: float, standard deviation value of purchases
--------------------------------------------------------------------------------

Returns
--------------------------------------------------------------------------------
pd.Series() with purchases values
--------------------------------------------------------------------------------
"""
        users_ = users_with_type_of_communication.set_index('id').loc[
            self.user_info['id'], ['communication']]
        prob_purch = self.daily_prob_of_purchase[self._zero_communication_value
                                                ].copy()
        for x in self.daily_prob_of_purchase.keys():
            ind = users_['communication'] == x
            prob_purch[ind] = self.daily_prob_of_purchase[x][ind]
        users_['purchases'] = (
            np.random.uniform(0, 1, size=len(prob_purch)) <= prob_purch)*1
        if add_purchases_value:
            mu = np.log(check_median)
            sigma = np.log((1 + (1+4*1*check_std**2/np.exp(2*mu))**0.5)/2)**0.5
            users_.loc[users_['purchases'] == 1,
                       'purchases'] = np.random.lognormal(
                mean=mu, sigma=sigma, size=np.sum(users_['purchases']))
        return users_with_type_of_communication['id'].map(users_['purchases'])