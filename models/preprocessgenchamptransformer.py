from dhi.dsmatch.sklearnmodeling.models.custombase import CustomTransformer

class PreProcessGenChampTransformer(CustomTransformer):
    
    def cur_desired_title(self,row):
        """We do not have enough current titles that is why we are using desired title in the palce of empty current title, In 
        this fuction 
        we are using below conditions:
        -  Keep if job title is same as current title
        - Else keep job title is same as desired title
        - Else current title is empty keep desired title
        - combined current and desited title with, keeping the unique

        Args: string
        Returns:string
        """
        if row['job_title']==row['current_title']: # if job title is same as current title keep this 
            row['current_title'] = row['job_title']
        if row['job_title'] in row['desired_title']:# else keep desired titles if it is in the job titles.
            row['current_title'] = row['job_title']  
        if row['current_title']=='':# if current title is empty or not present keep desired titles
            row['current_title'] = row['desired_title']
        row['current_title'] = set((row['current_title'].split('`'))+(row['desired_title'].split('`')))# combining current and 
                                                                                    #desured titles
        row['current_title'] = "::".join(row['current_title'])#join it with  ::.
        return row
    
    def clean_ready(self,df):
        """ Cleaning the job skills and profile skills
        """
        for c in ['job_skills', 'profile_skills']:
            df[c] = df[c].str.lower()# lowercase
            df[c] = df[c].str.split('::')# split them from ::
            df[c] = df[c].apply(lambda x: [s.strip() for s in x if s!=''])# removing empty strings
        for c in ['description_skills', 'resume_skills']:# these are Burning glass. parsed skills 
            df[c] = df[c].apply(eval)# The parsed skills comes in "['skill']" format so we convert into list of skills
            df[c] = df[c].apply(lambda x : [s.lower() for s in x])# lowercase the skills inside a list
        return df
    def transform(self,X, **kwargs):
        X = X.fillna('').astype(str).apply(lambda x: x.str.lower())
        X = X.apply(self.cur_desired_title, axis=1)
        X = self.clean_ready(X)
        return X
####################################################################################################################

class TitleMappingTransformer(CustomTransformer):
    """Combining job skills(user defined), Skills related to the titles and BG parsed skill. 
    Args: Dataframe of list[str]
    Returns: DataFramelist[str]
    """
    def comb_job_title_skills_bg(self,row):
        #combining job_skills a skills associated to job titles
        row['job_title_and_skills']= list(set(row['job_skills'] + row['job_title_to_skills'] + row['description_skills']))
        
        # combining profile_skills a skills associated to current titles  and previous title
        row['profile_title_and_skills']= list(set(row['profile_skills'] + row['current_title_to_skills'] + row['resume_skills'])) 
        return row
    def transform(self,X, **kwargs):
        # renameing some of the columns for better undestanding the columns
        X = X.rename(columns = {'dh_j_concatenated_titles': 'job_title_to_skills'
               ,'dh_r_concatenated_titles' : 'current_title_to_skills'
               ,'dh_p_concatenated_titles' : 'previous_title_to_skills'
               ,'dh_cur_pre_titles'        : 'concat_cur_pre_title_to_skills'})
        X = X.apply(self.comb_job_title_skills_bg, axis=1)
        return X
    
