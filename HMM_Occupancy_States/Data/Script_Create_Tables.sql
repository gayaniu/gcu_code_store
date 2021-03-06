USE [HMM_HomeStates]
GO
/****** Object:  Table [dbo].[All_Devices]    Script Date: 10/9/2020 12:39:39 PM ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO
CREATE TABLE [dbo].[All_Devices](
	[id] [int] NOT NULL,
	[name] [varchar](max) NULL,
	[components] [varchar](max) NULL,
	[roles] [varchar](max) NULL,
	[space_id] [int] NULL,
	[created_at] [datetime] NULL,
	[updated_at] [datetime] NULL,
	[domoticz_component_ids] [varchar](max) NULL,
	[provider] [varchar](max) NULL,
	[provider_id] [varchar](max) NULL,
	[space] [varchar](max) NULL,
	[component_states] [varchar](max) NULL,
	[Facility_No] [int] NOT NULL,
PRIMARY KEY CLUSTERED 
(
	[id] ASC,
	[Facility_No] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, IGNORE_DUP_KEY = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON) ON [PRIMARY]
) ON [PRIMARY] TEXTIMAGE_ON [PRIMARY]

GO
/****** Object:  Table [dbo].[Devices]    Script Date: 10/9/2020 12:39:39 PM ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO
CREATE TABLE [dbo].[Devices](
	[id] [int] NOT NULL,
	[name] [varchar](100) NULL,
	[components] [varchar](max) NULL,
	[roles] [varchar](100) NULL,
	[space_id] [int] NULL,
	[created_at] [varchar](100) NULL,
	[updated_at] [varchar](100) NULL,
	[domoticz_component_ids] [varchar](100) NULL,
	[provider] [varchar](100) NULL,
	[provider_id] [varchar](100) NULL,
PRIMARY KEY CLUSTERED 
(
	[id] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, IGNORE_DUP_KEY = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON) ON [PRIMARY]
) ON [PRIMARY] TEXTIMAGE_ON [PRIMARY]

GO
/****** Object:  Table [dbo].[Events]    Script Date: 10/9/2020 12:39:39 PM ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO
CREATE TABLE [dbo].[Events](
	[id] [int] NOT NULL,
	[event_id] [int] NULL,
	[occurred_at] [varchar](100) NULL,
	[event_value] [varchar](50) NULL,
	[sensor_type] [varchar](50) NULL,
	[sensor_id] [int] NULL,
	[space_id] [int] NULL,
	[event_date] [varchar](50) NULL,
PRIMARY KEY CLUSTERED 
(
	[id] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, IGNORE_DUP_KEY = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON) ON [PRIMARY]
) ON [PRIMARY]

GO
/****** Object:  Table [dbo].[Full_Events]    Script Date: 10/9/2020 12:39:39 PM ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO
CREATE TABLE [dbo].[Full_Events](
	[id] [int] NOT NULL,
	[space_id] [int] NULL,
	[provider_id] [int] NULL,
	[name] [varchar](100) NULL,
	[space] [varchar](50) NULL,
	[type] [varchar](50) NULL,
	[state] [varchar](50) NULL,
	[event_date] [varchar](50) NULL,
	[Facility_No] [int] NOT NULL,
 CONSTRAINT [PK__Full_Eve__3213E83F58594084] PRIMARY KEY CLUSTERED 
(
	[id] ASC,
	[Facility_No] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, IGNORE_DUP_KEY = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON) ON [PRIMARY]
) ON [PRIMARY]

GO
/****** Object:  Table [dbo].[Hmm_Results]    Script Date: 10/9/2020 12:39:39 PM ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO
CREATE TABLE [dbo].[Hmm_Results](
	[Facility_No] [int] NOT NULL,
	[id] [int] NOT NULL,
	[Observation] [varchar](50) NULL,
	[hmm_state] [varchar](50) NULL,
	[Prob_Occupied_Active] [float] NULL,
	[Prob_Occupied_Inactive] [float] NULL,
	[Prob_Unoccupied] [float] NULL,
	[Prob_Unknown] [float] NULL,
	[Val_Result] [varchar](max) NULL,
	[Duration_min] [int] NULL,
	[Accuracy] [int] NULL,
 CONSTRAINT [PK_hmm_results] PRIMARY KEY CLUSTERED 
(
	[Facility_No] ASC,
	[id] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, IGNORE_DUP_KEY = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON) ON [PRIMARY]
) ON [PRIMARY] TEXTIMAGE_ON [PRIMARY]

GO
/****** Object:  Table [dbo].[Lable_Results]    Script Date: 10/9/2020 12:39:39 PM ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO
CREATE TABLE [dbo].[Lable_Results](
	[Facility_No] [int] NOT NULL,
	[id] [int] NOT NULL,
	[space_id] [int] NULL,
	[space] [varchar](100) NULL,
	[type] [varchar](100) NULL,
	[state] [varchar](100) NULL,
	[pre_state] [varchar](100) NULL,
	[from] [datetime] NULL,
	[to] [datetime] NULL,
	[Time_Gap_Seconds] [int] NULL,
	[Time_Gap_Minutes] [int] NULL,
	[devices] [varchar](max) NULL,
	[Device_Roles] [varchar](max) NULL,
	[Neighbour_space_Id] [varchar](max) NULL,
	[First_label] [varchar](100) NULL,
 CONSTRAINT [pk_constraint] PRIMARY KEY CLUSTERED 
(
	[Facility_No] ASC,
	[id] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, IGNORE_DUP_KEY = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON) ON [PRIMARY]
) ON [PRIMARY] TEXTIMAGE_ON [PRIMARY]

GO
/****** Object:  Table [dbo].[Master_Space_Data]    Script Date: 10/9/2020 12:39:39 PM ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO
CREATE TABLE [dbo].[Master_Space_Data](
	[id] [int] NOT NULL,
	[name] [varchar](max) NULL,
	[kind] [varchar](max) NULL,
	[created_at] [datetime] NULL,
	[updated_at] [datetime] NULL,
	[transit_points] [varchar](max) NULL,
	[devices] [varchar](max) NULL,
	[transit_points_id] [int] NULL,
	[device_id] [int] NULL,
	[Facility_No] [int] NOT NULL,
PRIMARY KEY CLUSTERED 
(
	[id] ASC,
	[Facility_No] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, IGNORE_DUP_KEY = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON) ON [PRIMARY]
) ON [PRIMARY] TEXTIMAGE_ON [PRIMARY]

GO
/****** Object:  Table [dbo].[Space_Data]    Script Date: 10/9/2020 12:39:39 PM ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO
CREATE TABLE [dbo].[Space_Data](
	[id] [int] NOT NULL,
	[name] [varchar](max) NULL,
	[kind] [varchar](max) NULL,
	[created_at] [datetime] NULL,
	[updated_at] [datetime] NULL,
	[transit_points] [varchar](max) NULL,
	[devices] [varchar](max) NULL,
	[transit_points_id] [int] NULL,
	[device_id] [int] NULL,
PRIMARY KEY CLUSTERED 
(
	[id] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, IGNORE_DUP_KEY = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON) ON [PRIMARY]
) ON [PRIMARY] TEXTIMAGE_ON [PRIMARY]

GO
/****** Object:  Table [dbo].[Transit_Points]    Script Date: 10/9/2020 12:39:39 PM ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO
CREATE TABLE [dbo].[Transit_Points](
	[id] [int] NOT NULL,
	[device_id] [int] NULL,
	[space_id] [int] NULL,
	[neighbour_space_id] [int] NULL,
	[created_at] [varchar](100) NULL,
	[updated_at] [varchar](100) NULL,
	[Facility_No] [int] NOT NULL,
PRIMARY KEY CLUSTERED 
(
	[id] ASC,
	[Facility_No] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, IGNORE_DUP_KEY = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON) ON [PRIMARY]
) ON [PRIMARY]

GO
/****** Object:  Table [dbo].[Unlabled_Data]    Script Date: 10/9/2020 12:39:39 PM ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO
CREATE TABLE [dbo].[Unlabled_Data](
	[id] [int] NOT NULL,
	[space_id] [int] NULL,
	[space] [varchar](100) NULL,
	[type] [varchar](100) NULL,
	[state] [varchar](100) NULL,
	[FROM] [datetime] NULL,
	[To] [datetime] NULL,
	[Time_Gap_Seconds] [int] NULL,
	[Time_Gap_Minutes] [int] NULL,
	[Devices] [varchar](1000) NULL,
	[Device_Roles] [varchar](max) NULL,
	[Neighbouring_spaces] [varchar](1000) NULL,
PRIMARY KEY CLUSTERED 
(
	[id] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, IGNORE_DUP_KEY = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON) ON [PRIMARY]
) ON [PRIMARY] TEXTIMAGE_ON [PRIMARY]

GO
/****** Object:  Table [dbo].[Validation_Facility_data]    Script Date: 10/9/2020 12:39:39 PM ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO
CREATE TABLE [dbo].[Validation_Facility_data](
	[Activity] [varchar](200) NULL,
	[Day_] [varchar](100) NULL,
	[Space_] [varchar](100) NULL,
	[From_] [time](7) NULL,
	[To_] [time](7) NULL,
	[Facility_No] [int] NULL
) ON [PRIMARY]

GO
/****** Object:  StoredProcedure [dbo].[Get_Hmm_Validation_Statistics]    Script Date: 10/9/2020 12:39:39 PM ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO
-- =============================================
-- Author:		<Author,,Name>
-- Create date: <Create Date,,>
-- Description:	<Description,,>
-- =============================================
--EXEC [dbo].[Get_Hmm_Validation_Statistics] 
CREATE PROCEDURE [dbo].[Get_Hmm_Validation_Statistics]
	@Date Datetime =  null
	,@Space VARCHAR(20) = NULL
AS
BEGIN
	-- SET NOCOUNT ON added to prevent extra result sets from
	-- interfering with SELECT statements.
	SET NOCOUNT ON;

	Declare @sum AS INT 
	Select @sum = COUNT( A.[Accuracy])  from   [dbo].[Hmm_Results] A

	
	SELECT CASE WHEN A.Accuracy = 0 THEN 'Incorrect' 
				WHEN A.Accuracy = 1  THEN 'Correct'
				WHEN A.Accuracy = 2 THEN 'Incorrect or Correct' 
				ELSE 'N/A' END AS Category,
		   COUNT(A.Accuracy) as [Event count],
		  COUNT(A.Accuracy) * 100.0 / @sum as [percent]
	FROM    [dbo].[Hmm_Results] A 
	INNER JOIN Full_Events B ON A.Facility_No =B.Facility_No AND A.id = B.id
	group by (A.Accuracy)

	
	
	
END

GO
/****** Object:  StoredProcedure [dbo].[SP_Get_Occupancy_Data_By_Space]    Script Date: 10/9/2020 12:39:39 PM ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO
-- =============================================
-- Author: GAYANI UDAWATTA 
-- Create date: <Create Date,,>
-- Description: Get the labeled data for Viterbi Algorithm
-- =============================================
--exec [dbo].[SP_Get_Occupancy_Data_By_Space] 'kitchen'
CREATE PROCEDURE [dbo].[SP_Get_Occupancy_Data_By_Space] 
	@Facility_No AS INT
	 ,@Space VARCHAR(100) = NULL 
AS
BEGIN
	-- SET NOCOUNT ON added to prevent extra result sets from
	-- interfering with SELECT statements.
	SET NOCOUNT ON;
	-- for the whole house-- 2 doors for transists  
	-- when we consider the bedroom, there can be only one transit point

    -- Inserting data to Temp table by converting event date in to Australian date time from UTC datetime
	IF (@Space = 'ALL')
	BEGIN
	 SET @Space =NULL
	END
    SELECT 
		 FE.[id]
		, FE.[space_id]
		, FE.[space]
		, FE.[type] 
		, FE.[state]
		, FE.[event_date]
		, DATEADD(SECOND, DATEDIFF(SECOND, GETUTCDATE(), GETDATE()), FE.[event_date])   AS  [AUS Event_Date] 		
	INTO #temp_Full_Events
	FROM [dbo].[Full_Events] FE
		INNER JOIN  [Master_Space_Data] SD  
			ON FE.[space] =  SD.[kind]
	--REMOVING THE CONTACT SENSOR RECORDS FOR FRIDGE, MICROWAVE AND PANTRY DOORS AS THEY ARE ASSUMED AS ANOMALIES -- 16 SENSOR MIGT RECORDED THE OCCUPANCY
	WHERE 	 FE.[Space] =  CASE WHEN @Space IS NULL THEN [Space] ELSE @Space END	
			 AND FE.[space_id] NOT IN  	 (30,31,32,33)	
			--AND ((@Space IS NULL AND FE.space_id NOT IN (31,32,33) ) OR (@Space ='kitchen'  AND FE.space_id NOT IN  (31,32,33) )) 	  
	ORDER BY   FE.[id] 	



	SELECT 
		FE.id
		, FE.space_id
		, CASE WHEN @Space IS NULL THEN 'ALL' ELSE FE.[space] END AS [space]
		, FE.[type] 
		, FE.[state]
		, FE.[event_date]
		, FE.[AUS Event_Date] 
		,ROW_NUMBER() OVER ( PARTITION BY  FE.[AUS Event_Date] 	ORDER BY  FE.[AUS Event_Date] ) AS [row] 
	INTO #temp_All_Events
	FROM #temp_Full_Events FE
	WHERE CAST(FE.[AUS Event_Date] AS DATE) ='2016-12-21' 

	SELECT 
	 id
	,[space]
	,space_id
	,[type] 
	,LAG([state],1) OVER (PARTITION BY  [space],[type]  ORDER BY [AUS Event_Date]  ASC  ) AS  [Pre_state]
	,[state]
	,[event_date]		
	,[AUS Event_date]
	,[row]
	,LAG([state],1) OVER (PARTITION BY  [space],[type] ORDER BY [AUS Event_Date]  ASC  ) +'-'+[state] AS [Status] 
	INTO  #Temp_inter_data
	FROM #temp_All_Events
	WHERE [row] = 1
	ORDER BY [AUS Event_Date] ASC 

	

	SELECT 
	 id
	,[space]
	,space_id
	,[type] 
	,[Pre_state]
	, [state]
	, [event_date]		
	, LAG([AUS Event_date],1)   OVER( PARTITION BY  [space]
    ORDER BY [AUS Event_Date]  ASC )
		AS [From] 
		, [AUS Event_Date] AS [To] 	
		,DATEDIFF( SECOND, LAG([AUS Event_date],1)   OVER( PARTITION BY  [space]
    ORDER BY [AUS Event_Date]  ASC ) , [AUS Event_Date] )  AS  [Time_Gap_SS]   	 
		 , DATEDIFF( MINUTE, LAG([AUS Event_date],1)   OVER( PARTITION BY  [space]
    ORDER BY [AUS Event_Date]  ASC ) , [AUS Event_Date] )  AS  [Time_Gap_MIN] 
	, (SELECT AD.[roles] from  All_Devices  AD WHERE AD.[id] = #Temp_inter_data.[space_id]) AS  [Device_Role] 
		, (select MSD.[name] from Master_Space_Data MSD  WHERE #Temp_inter_data.[space] = MSD.[kind]  ) AS [Spaceid]	
	INTO  #Temp_final_data
	FROM #Temp_inter_data
	WHERE [row] = 1
	AND [Pre_state] <> [state] 
	--AND [Status] not in ( 'closed-inactive' , 'open-active', 'inactive-closed', 'active-open') 
	ORDER BY [AUS Event_Date] ASC 	



	---------------	--1. labeling -------------------------------------------------------------------------------------------------

	--select * from #Temp_final_data 

	SELECT 
		[TD].[id]
		, [TD].[space]
		, [TD].[space_id] as [Device_id]
		, [TD].[type] 
		, [TD].[Pre_state]
		, [TD].[state]
		, [TD].[event_date]		
		, [TD].[From] 
		, [TD].[To] 	
		, [TD].[Time_Gap_SS]   	 
		, [TD].[Time_Gap_MIN] 
		, [TD].Device_Role
		,[TD].[Spaceid] AS [Space_id]			 
		,CASE	WHEN ( (@Space IS NOT NULL AND TD.Device_Role like '%transit%')	AND  /*( [TD].[Pre_state] ='open'	OR [TD].[Pre_state] ='active' )			AND */( [TD].[Time_Gap_SS]    > 0		/*AND  [TD].[Time_Gap_SS]    <= 240*/)  )		THEN 'Transit'
				WHEN ( (@Space IS NULL AND TD.Device_Role like '%transit%')				AND ( [TD].[Time_Gap_SS]    > 0		))		THEN 'Transit'
				WHEN (  ([TD].[Pre_state] ='active'			OR		[TD].[Pre_state] ='open' )	AND ( [TD].[Time_Gap_SS]   > 0	AND [TD].[Time_Gap_SS]    <=300  ) )		THEN 'Active' --'Short_Active'
				WHEN ( ( [TD].[Pre_state] ='active'			OR		[TD].[Pre_state] ='open')	AND  [TD].[Time_Gap_SS]    >300    )											THEN 'Active'		 
				WHEN (  ([TD].[Pre_state] ='inactive'		OR		[TD].[Pre_state] ='closed')	AND ( [TD].[Time_Gap_SS]   >0		AND [TD].[Time_Gap_SS]    <= 300   ) )		THEN 'Short_Inactive'
				WHEN ( (  [TD].[Pre_state] ='inactive'		OR		[TD].[Pre_state] ='closed' )AND ( [TD].[Time_Gap_SS]   >300    ) )											THEN 'Long_Inactive'
			ELSE 'TR_LongInactive'
		END AS First_Label
	INTO #temp_first_labels
	FROM  	#Temp_final_data TD
	WHERE Time_Gap_SS  IS NOT NULL 
	AND [Time_Gap_SS] > 0 
	ORDER BY [From]  ASC

	
	--------------------------------------------------------------------------------------------------------------------------------------



	SELECT 
		[TD].[id]
		,[TD].[Space_id]	 AS [Space_id] 
		, [TD].[space]
		, [TD].[Device_id] as [Device Id]
		, [TD].[type] 
		--, [TD].[Pre_state]
		--, [TD].[state]
		--, [TD].[event_date]		
		, [TD].[From] 
		, [TD].[To] 	
		, [TD].[Time_Gap_SS]   	 
		, [TD].[Time_Gap_MIN] 
		--, [TD].[Devices] 
		, [TD].[Device_Role]
		--, [TD].[Neighbouring_Spaces] 
		, CASE WHEN ( [TD].[First_label] = 'Transit' AND  LEAD ([TD].[First_label],1)   OVER( PARTITION BY  [space] ORDER BY [TD].[From]  ) = 'Long_Inactive'  ) THEN 'TR_LongInactive'
		WHEN ( [TD].[First_label] = 'Long_Inactive' AND  LAG ([TD].[First_label],1)   OVER( PARTITION BY  [space]  ORDER BY [TD].[From] ) = 'Transit' ) THEN 'TR_LongInactive'
		ELSE TD.[First_label] 		END AS [First_label]		
	INTO #Temp_All
	FROM #temp_first_labels TD
	ORDER BY [TD].[From] ASC


	SELECT	
		[TD].[id]
		,[TD].[Space_id] 
		, [TD].[space]
		, [TD].[Device Id]
		, [TD].[type]				
		, [TD].[From] 
		, [TD].[To] 	
		, [TD].[Time_Gap_SS]   	 
		, [TD].[Time_Gap_MIN] 		
		, [TD].[Device_Role]	
		, [TD].[First_Label]
		, CASE 	WHEN TD.[First_label] = 'Transit'			THEN 0
				WHEN  TD.[First_label] = 'Active'		THEN 1			
				WHEN TD.[First_label] = 'Short_Inactive'	THEN 2
				WHEN TD.[First_label] = 'Long_Inactive'		THEN 3
				WHEN  [TD].[First_label] = 'TR_LongInactive' THEN 4
		ELSE 5  END
		AS [Encoding]
	
	FROM 

	#Temp_All [TD]

	DROP TABLE  #temp_All_Events
	DROP TABLE #Temp_final_data
	DROP TABLE #temp_Full_Events
	DROP TABLE #Temp_inter_data
	DROP TABLE #Temp_All

	DROP TABLE #temp_first_labels



END

GO
/****** Object:  StoredProcedure [dbo].[Sp_Validate_Hmm_Result]    Script Date: 10/9/2020 12:39:39 PM ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO
-- =============================================
-- Author:		<Author,,Name>
-- Create date: <Create Date,,>
-- Description:	<Description,,>
-- =============================================
-- EXEC [dbo].[Sp_Validate_Hmm_Result]
CREATE PROCEDURE [dbo].[Sp_Validate_Hmm_Result]
	@Space  VARCHAR(100) = NULL
AS
BEGIN
	-- SET NOCOUNT ON added to prevent extra result sets from
	-- interfering with SELECT statements.
	SET NOCOUNT ON;


	
select 

A.Facility_No 
,A.id
,Observation 
,hmm_state 

,CASE WHEN Prob_Occupied_Active >= Prob_Occupied_Inactive AND Prob_Occupied_Active >= Prob_Unoccupied AND Prob_Occupied_Active >= Prob_Unknown THEN convert(numeric(18,18),convert(real,Prob_Occupied_Active))
            WHEN Prob_Occupied_Inactive >= Prob_Occupied_Active AND Prob_Occupied_Inactive >= Prob_Unoccupied AND Prob_Occupied_Inactive >= Prob_Unknown THEN convert(numeric(18,18),convert(real,Prob_Occupied_Inactive))
            WHEN Prob_Unoccupied >= Prob_Occupied_Active AND Prob_Unoccupied >= Prob_Occupied_Inactive AND Prob_Unoccupied >= Prob_Unknown THEN convert(numeric(18,18),convert(real,Prob_Unoccupied))
            WHEN Prob_Unknown >= Prob_Occupied_Active AND Prob_Unknown >= Prob_Occupied_Inactive AND Prob_Unknown >= Prob_Unoccupied THEN convert(numeric(18,18),convert(real,Prob_Unknown))
        END As Probability
,B.[space]
,B.event_date 
, DATEADD(SECOND, DATEDIFF(SECOND, GETUTCDATE(), GETDATE()), B.[event_date])  AS EventDate
,cast( CONVERT(VARCHAR(8),DATEADD(SECOND, DATEDIFF(SECOND, GETUTCDATE(), GETDATE()), B.[event_date]),108) AS TIME) AS [Occured_Time]
,Val_Result
,Duration_min
, Accuracy 
into #temp_OA
from    [dbo].[Hmm_Results] A INNER JOIN Full_Events B ON A.Facility_No =B.Facility_No AND A.id = B.id
where hmm_state IN ( 'OCCUPIED_ACTIVE' ,'OCCUPIED_INACTIVE' )  OR Observation in ( 'Active' ,'Short_Inactive' , 'Long_Inactive')



UPDATE Hmm_Results
SET Val_Result = 'sleeping' , Accuracy = 1
where [id] in ( select [id] from  #temp_OA
where Occured_Time between '22:00:00.0000000' and '00:59:00.0000000'
OR  Occured_Time between '01:00:00.0000000' and '06:59:00.0000000'
and [space] =  'master_bedroom'  and Val_Result is  null)






UPDATE Hmm_Results
SET Val_Result = 'Wake Up' , Accuracy = 1
where [id] in (
select [id] from  #temp_OA
where Occured_Time between '05:00:00.0000000' and '08:30:00.0000000'
and [space] =  'master_bedroom' and Val_Result is  null)




--UPDATING OTHER RECORDS FOR MASTER BED ROOM
UPDATE Hmm_Results SET Val_Result = 'No Activity'  , Accuracy = 1
WHERE [id] IN (
SELECT HR.[id]  FROM  Hmm_Results HR 
INNER JOIN Full_Events  FE ON HR.[id]  = FE.[id] 
WHERE   FE.[space] ='master_bedroom' And  HR.[hmm_state] IN ('UNOCCUPIED', 'UNKNOWN') AND  HR.[id] NOT IN   ( select [id] from  #temp_OA
where ( Occured_Time between '22:00:00.0000000' and '00:59:00.0000000'
OR  Occured_Time between '01:00:00.0000000' and '06:59:00.0000000' )    AND [space] ='master_bedroom'  and Val_Result is   null  ) 
)


--Breakfast
UPDATE Hmm_Results
SET Val_Result = 'Breakfast' , Accuracy = 1
where [id] in (
select [id] from  #temp_OA
where Occured_Time between '08:00:00.0000000' and '10:30:00.0000000'
and [space] =  'kitchen'  and Val_Result is  null )

--lunch cooking
UPDATE Hmm_Results
SET Val_Result = 'Cooking- Lunch' , Accuracy = 1
where [id] in (
select [id] from  #temp_OA
where Occured_Time between '12:00:00.0000000' and '13:30:00.0000000'
and  Val_Result is  null and  [space] =  'kitchen'  )





-- UPDATING UNOCCUPANCY IN KITCHEN

UPDATE Hmm_Results SET Val_Result = 'No Activity' , Accuracy = 1
WHERE [id] IN ( 
SELECT HR.[id] FROM  Hmm_Results HR 
INNER JOIN Full_Events  FE ON HR.[id]  = FE.[id] 
WHERE   FE.[space] ='kitchen' And  HR.[hmm_state] IN ('UNOCCUPIED', 'UNKNOWN') AND  HR.[id] NOT IN   ( select [id] from  #temp_OA
where ( ( Occured_Time between '07:00:00.0000000' and '08:30:00.0000000' ) OR (Occured_Time between '08:00:00.0000000' and '10:30:00.0000000') OR (Occured_Time between '12:30:00.0000000' and '13:00:00.0000000') OR ( Occured_Time between '19:00:00.0000000' and '19:30:00.0000000'))   AND [space] ='kitchen'  and Val_Result is   null  ) 

)



-- Morning Bathroom
UPDATE Hmm_Results
SET Val_Result = 'Morning_bathroom' , Accuracy = 1
where [id] in (
select [id] from  #temp_OA
where Occured_Time between '06:00:00.0000000' and '08:30:00.0000000'
and [space] =  'bathroom' and Val_Result is  null )


--Midnight_bathroom
UPDATE Hmm_Results
SET Val_Result = 'Midnight_bathroom' , Accuracy = 1
where [id] in (
select [id] from  #temp_OA
where Occured_Time between '01:00:00.0000000' and '04:30:00.0000000'
and [space] =  'bathroom'  and Val_Result is  null )


--Toilet visits
UPDATE Hmm_Results
SET Val_Result = 'Toilet visits' , Accuracy = 1
where [id] in (
select [id] from  #temp_OA
where  [space] =  'bathroom'  and Val_Result is  null AND Duration_min between 1 and 6)
--Showering
UPDATE Hmm_Results
SET Val_Result = 'Showering' , Accuracy = 1
where [id] in (
select [id] from  #temp_OA
where  [space] =  'bathroom'  and Val_Result is  null AND Duration_min between 1 and 30)

-- UPDATING BATHROOM UNOCCUPANCY
UPDATE Hmm_Results
SET Val_Result = 'No Activity' , Accuracy = 1
WHERE [id] IN (
select HR.[id] from Hmm_Results hr inner join Full_Events  fe on hr.[id] = fe.[id]
where fe.[space] ='bathroom'  and Val_Result is  null )





UPDATE Hmm_Results
SET Val_Result = 'Outing' , Accuracy = 1
where [id] in (
select [id] from  #temp_OA
where Occured_Time between '07:00:00.0000000' and '14:30:00.0000000'
and  Val_Result is null and FORMAT(cast(EventDate as datetime2) ,'dddd' ) = 'Tuesday'   )

update Hmm_Results set Val_Result ='No Activity' , Accuracy = 1
where  id in (
SELECT hr.id FROM Hmm_Results hr inner join  Full_Events fr on hr.[id] = fr.[id]  WHERE  hr.[hmm_state] = 'UNOCCUPIED' AND FORMAT(cast( DATEADD(SECOND, DATEDIFF(SECOND, GETUTCDATE(), GETDATE()), fr.[event_date]) as datetime2) ,'dddd' ) = 'Tuesday'  AND hr.[id] not in  (
select [id] from  #temp_OA
where Occured_Time between '07:00:00.0000000' and '14:30:00.0000000'
and  Val_Result is null and FORMAT(cast(EventDate as datetime2) ,'dddd' ) = 'Tuesday'   ) 
)  

update Hmm_Results set Val_Result ='No Activity' , Accuracy = 0
where  id in (

SELECT hr.id FROM Hmm_Results hr inner join  Full_Events fr on hr.[id] = fr.[id]  WHERE  hr.[hmm_state] <> 'UNOCCUPIED' AND FORMAT(cast( DATEADD(SECOND, DATEDIFF(SECOND, GETUTCDATE(), GETDATE()), fr.[event_date]) as datetime2) ,'dddd' ) = 'Tuesday'  AND hr.[id] not in  (
select [id] from  #temp_OA
where Occured_Time between '07:00:00.0000000' and '14:30:00.0000000'
and  Val_Result is null and FORMAT(cast(EventDate as datetime2) ,'dddd' ) = 'Tuesday'   ) 
)  




UPDATE Hmm_Results
SET Val_Result = 'Visitor' , Accuracy = 1
where [id] in (
select [id] from  #temp_OA
where Occured_Time between '12:00:00.0000000' and '19:30:00.0000000'
and  Val_Result is  null and FORMAT(cast(EventDate as datetime2) ,'dddd' ) = 'Wednesday'   )


UPDATE Hmm_Results
SET Val_Result = 'Cleaning' , Accuracy = 1
where [id] in (
select [id] from  #temp_OA
where Occured_Time between '08:00:00.0000000' and '09:30:00.0000000'
and  Val_Result is  null and FORMAT(cast(EventDate as datetime2) ,'dddd' ) = 'Wednesday'   )






UPDATE Hmm_Results
SET Val_Result = 'Watching TV' , Accuracy = 1
where [id] in (
select [id] from  #temp_OA
where Occured_Time between '12:00:00.0000000' and '17:30:00.0000000'
and  Val_Result is  null and  [space] =  'lounge'  )


--lunch 
UPDATE Hmm_Results
SET Val_Result = 'Lunch' , Accuracy = 1
where [id] in (
select [id] from  #temp_OA
where Occured_Time between '13:00:00.0000000' and '14:00:00.0000000'
and  Val_Result is  null and  ( [space]   = 'lliving_room' or   [space]   = 'lounge')  )


--dinner
UPDATE Hmm_Results
SET Val_Result = 'Dinner' , Accuracy = 1
where [id] in (
select [id] from  #temp_OA
where Occured_Time between '19:30:00.0000000' and '20:00:00.0000000'
and  Val_Result is  null and  ( [space]   = 'lliving_room' or   [space]   = 'lounge')  )


--Washing clothes
UPDATE Hmm_Results
SET Val_Result = 'Washing Clothes' , Accuracy = 1
where [id] in (
select [id] from  #temp_OA
where Occured_Time between '09:00:00.0000000' and '10:00:00.0000000'
and  Val_Result is  null and   [space]   = 'laundry '   )

--Medication
UPDATE Hmm_Results
SET Val_Result = 'Medication' , Accuracy = 1
where [id] in (
select [id] from  #temp_OA
where Occured_Time between '08:30:00.0000000' and '09:00:00.0000000'
and  Val_Result is  null   )

--Medication
UPDATE Hmm_Results
SET Val_Result = 'Medication' , Accuracy = 1
where [id] in (
select [id] from  #temp_OA
where Occured_Time between '21:00:00.0000000' and '21:30:00.0000000'
and  Val_Result is  null   )




--Reading
UPDATE Hmm_Results
SET Val_Result = 'Reading', Accuracy = 1
where [id] in (
select [id] from  #temp_OA
where  [space] =  'office  '  and Val_Result is  null AND Duration_min between 1 and 60)

UPDATE Hmm_Results
SET Val_Result = 'Computer' , Accuracy = 1
where [id] in (
select [id] from  #temp_OA
where  [space] =  'office  '  and Val_Result is  null AND Duration_min> 60)

select 
A.Facility_No 
,A.id
,Observation 
,hmm_state 

,CASE WHEN Prob_Occupied_Active >= Prob_Occupied_Inactive AND Prob_Occupied_Active >= Prob_Unoccupied AND Prob_Occupied_Active >= Prob_Unknown THEN convert(numeric(18,18),convert(real,Prob_Occupied_Active))
            WHEN Prob_Occupied_Inactive >= Prob_Occupied_Active AND Prob_Occupied_Inactive >= Prob_Unoccupied AND Prob_Occupied_Inactive >= Prob_Unknown THEN convert(numeric(18,18),convert(real,Prob_Occupied_Inactive))
            WHEN Prob_Unoccupied >= Prob_Occupied_Active AND Prob_Unoccupied >= Prob_Occupied_Inactive AND Prob_Unoccupied >= Prob_Unknown THEN convert(numeric(18,18),convert(real,Prob_Unoccupied))
            WHEN Prob_Unknown >= Prob_Occupied_Active AND Prob_Unknown >= Prob_Occupied_Inactive AND Prob_Unknown >= Prob_Unoccupied THEN convert(numeric(18,18),convert(real,Prob_Unknown))
        END As Probability
,B.[space]
,B.event_date 
, DATEADD(SECOND, DATEDIFF(SECOND, GETUTCDATE(), GETDATE()), B.[event_date])  AS EventDate
,cast( CONVERT(VARCHAR(8),DATEADD(SECOND, DATEDIFF(SECOND, GETUTCDATE(), GETDATE()), B.[event_date]),108) AS TIME) AS [Occured_Time]
,Val_Result
,Duration_min
, Accuracy
into #temp_rem
from    [dbo].[Hmm_Results] A INNER JOIN Full_Events B ON A.Facility_No =B.Facility_No AND A.id = B.id 
 where Val_Result is  null

  
  UPDATE Hmm_Results 
  SET Val_Result = 'No Activity' , Accuracy = 0 
  WHERE [id] in ( 
  select [id] from #temp_rem 
  where [space] Not IN (  'master_bedroom', 'bathroom') AND  ( ( Occured_Time between  '22:00:00.0000000' and '00:59:00.0000000' ) OR   ( Occured_Time between  '01:00:00.0000000' and '06:59:00.0000000' ))
   and Val_Result is  null
   and [hmm_state] IN  ('OCCUPED_ACTIVE', 'OCCUPIED_INACTIVE' , 'TRANSIT' )
   )


   UPDATE Hmm_Results SET Val_Result = 'Activity' , Accuracy = 2  -- worng/correct
   WHERE [id] in (
                     SELECT [id] FROM #temp_OA where Val_Result  is null and   [hmm_state] IN  ('OCCUPIED_ACTIVE')
					 
					 ) 
UPDATE Hmm_Results
SET Val_Result = 'Incorrect'  , Accuracy = 0 
where Observation = 'Transit' and hmm_state = 'OCCUPIED_INACTIVE'


UPDATE Hmm_Results 
SET Val_Result = 'Correct' , Accuracy = 1
WHERE Val_Result is null and  Observation = 'Transit' and hmm_state = 'UNKNOWN'

UPDATE Hmm_Results 
SET Val_Result = 'No Activity' , Accuracy = 2  -- worng/correct
WHERE Val_Result is null and Observation = 'TR-Long_Inactive' and hmm_state = 'UNOCCUPIED'

UPDATE Hmm_Results 
SET Val_Result = 'No Activity' , Accuracy = 2  -- worng/correct
WHERE  [id] in ( select [id] from #temp_rem 
where [space] = 'lounge' and  Occured_Time between '10:00:00.0000000' and '12:00:00.0000000'
)

UPDATE Hmm_Results 
SET Val_Result = 'Correct' , Accuracy = 1
WHERE Val_Result is null and  Observation = 'Short_Inactive' and hmm_state = 'OCCUPIED_INACTIVE'

Select 
A.Facility_No 
,A.id
,Observation 
,hmm_state 

,CASE WHEN Prob_Occupied_Active >= Prob_Occupied_Inactive AND Prob_Occupied_Active >= Prob_Unoccupied AND Prob_Occupied_Active >= Prob_Unknown THEN convert(numeric(18,5),convert(real,Prob_Occupied_Active))
            WHEN Prob_Occupied_Inactive >= Prob_Occupied_Active AND Prob_Occupied_Inactive >= Prob_Unoccupied AND Prob_Occupied_Inactive >= Prob_Unknown THEN convert(numeric(18,5),convert(real,Prob_Occupied_Inactive))
            WHEN Prob_Unoccupied >= Prob_Occupied_Active AND Prob_Unoccupied >= Prob_Occupied_Inactive AND Prob_Unoccupied >= Prob_Unknown THEN convert(numeric(18,5),convert(real,Prob_Unoccupied))
            WHEN Prob_Unknown >= Prob_Occupied_Active AND Prob_Unknown >= Prob_Occupied_Inactive AND Prob_Unknown >= Prob_Unoccupied THEN convert(numeric(18,5),convert(real,Prob_Unknown))
        END As Probability
,B.[space]
,B.event_date 
, DATEADD(SECOND, DATEDIFF(SECOND, GETUTCDATE(), GETDATE()), B.[event_date])  AS EventDate
,cast( CONVERT(VARCHAR(8),DATEADD(SECOND, DATEDIFF(SECOND, GETUTCDATE(), GETDATE()), B.[event_date]),108) AS TIME) AS [Occured_Time]
,Val_Result
,Duration_min
, Accuracy 
FROM    [dbo].[Hmm_Results] A INNER JOIN Full_Events B ON A.Facility_No =B.Facility_No AND A.id = B.id



drop table #temp_OA
drop table #temp_rem


 








END

GO
