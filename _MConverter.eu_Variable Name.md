| **Variable Name**             | Retain or Drop | Brief justification for retention or dropping                                                                                       |
|-------------------------------|----------------|-------------------------------------------------------------------------------------------------------------------------------------|
| ID                            | Drop           | **This is not relevant to the target variable and may cause noise. This is because it is not applicable to the status prediction.** |
| Sex                           | Drop           | **This does not contain information relevant to the status prediction.**                                                            |
| Age                           | Drop           | **It is constant in most. Minimal variation to learn from. For example, age 40 in most.**                                           |
| Education Qualifications      | Retain         | **This is because the level of education helps predict social and financial stability.**                                            |
| Income                        | Retain         | **It aids in predicting the loan-to-income ratio.**                                                                                 |
| Home Ownership                | Retain         | **It helps determine whether the person has collateral and also their monthly liabilities or expenditures on housing.**             |
| Employment Length             | Retain         | **It is helpful to determine how long the person has been working, as well as the stability of their income stream.**               |
| Loan Intent                   | Retain         | **It helps to determine if there would be a return.**                                                                               |
| Loan Amount                   | Retain         | **Central variable for determining the loan approval status.**                                                                      |
| Loan Interest Rate            | Retain         | **Necessary for profit prediction.**                                                                                                |
| Loan-to-Income Ratio (LTI)    | Retain         | **Helps to predict the loan burden on income.**                                                                                     |
| Payment Default on File       | Retain         | **Necessary to determine past defaults on previous loans, to predict the probability of returns or loan repayment compliance.**     |
| Credit History Length         | Retain         | **Predict credit and risk behavior.**                                                                                               |
| Loan Approval Status          | Retain         | **This is the output (target) variable. Central prediction for test training valuation.**                                           |
| Maximum Loan Amount           | Retain         | **Determine the amount to be issued following an approval.**                                                                        |
| Credit Application Acceptance | Drop           | **Duplicate variable of loan approval status. It would introduce noise.**                                                           |

<table style="width:100%;">
<colgroup>
<col style="width: 15%" />
<col style="width: 24%" />
<col style="width: 32%" />
<col style="width: 27%" />
</colgroup>
<thead>
<tr class="header">
<th>Variable</th>
<th>Issue description</th>
<th>Proposed mitigation</th>
<th>Justification for used mitigation</th>
</tr>
</thead>
<tbody>
<tr class="odd">
<td>Education qualification</td>
<td>Whitespace in values e.g, “Unknown” and “Unknown ”</td>
<td><strong>Remove whitespace</strong></td>
<td><strong>It prevents duplication of class</strong></td>
</tr>
<tr class="even">
<td>Income</td>
<td>Wide range values and presence of outliers e.g, $9,600 to
$550,000</td>
<td><strong>Use a robust scaler for preprocessing</strong></td>
<td><strong>The use of median and Inter-quartile range (IQR) instead of
mean makes it resistant to outliers. In this case, the outliers may be
necessary for predicting large loan amounts.</strong></td>
</tr>
<tr class="odd">
<td>Employment Length</td>
<td><p>Spelling error (emplyment instead of employment).</p>
<p>The values range from 0 to 22 years, and this could suggest
outliers.</p></td>
<td><p><strong>Rename emplyment length to employment
length.</strong></p>
<p>Use a robust scaler for preprocessing.</p></td>
<td><p><strong>Fixes typo error and gives clarity.</strong></p>
<p><strong>Robust scaler handles outliers without removing
them.</strong></p></td>
</tr>
<tr class="even">
<td>Loan Intent</td>
<td>No spaces between words for two categories, e.g, HOMEIMPROVEMENT and
DEBTCONSOLIDATION</td>
<td><strong>Rename (clean and format) the categories by including a
space. Home improvement, Debt consolidation</strong></td>
<td><strong>It aids readability</strong></td>
</tr>
<tr class="odd">
<td>Loan Amount</td>
<td>Wide range of values and high loan amount requests</td>
<td><strong>Robust scaler for preprocessing</strong></td>
<td><strong>Handles outliers without removing them. These large amounts
may provide information and a relationship with the target
variable.</strong></td>
</tr>
<tr class="even">
<td>Loan Interest Rate</td>
<td>Unusually low to high rates between 5% to 20%</td>
<td><strong>Robust Scaler for scaling</strong></td>
<td><strong>The interest rate has a linear relationship with risk and is
a good predictor of approval status.</strong></td>
</tr>
<tr class="odd">
<td>Loan-to-Income Ratio (LTI)</td>
<td>Values range from 0.03 to 0.42</td>
<td><strong>Scale with Robust scaler</strong></td>
<td><strong>It is a significant risk indicator. It is also a domain
metric for loan approval.</strong></td>
</tr>
<tr class="even">
<td>Payment Default on File</td>
<td>Values are “Y” or “N”</td>
<td><strong>Encode the values in binary. “Y” to “1” and “N” to
“0”</strong></td>
<td><strong>It is simple and interpretable.</strong></td>
</tr>
<tr class="odd">
<td>Credit History Length</td>
<td>Values range between 3 to 30 years</td>
<td><strong>Scale with Robust scaler. Keep outliers.</strong></td>
<td><strong>It could determine the loan stability of
individuals.</strong></td>
</tr>
<tr class="even">
<td>Loan Approval Status</td>
<td><p>Inconsistent target variable (Approved, Declined, Reject).</p>
<p>Missing values.</p>
<p>Imbalanced class labels</p></td>
<td><p><strong>Rename (Clean and format) the variables.</strong></p>
<p>Drop rows containing missing values.</p></td>
<td><p><strong>Ensures consistent variables.</strong></p>
<p><strong>There is only one row containing missing values for loan
approval status.</strong></p></td>
</tr>
<tr class="odd">
<td>Maximum Loan Amount</td>
<td>Negative loan values</td>
<td><strong>Drop rows with negative values or change to absolute
values</strong></td>
<td><p><strong>Drop rows with negative values, as this may introduce
noise and lead to incorrect predictions.</strong></p>
<p><strong>Investigate if changing to absolute values would reduce the
outliers or pose a complication to the predictions.</strong></p></td>
</tr>
</tbody>
</table>
