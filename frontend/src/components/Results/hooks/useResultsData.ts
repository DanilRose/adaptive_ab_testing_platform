import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { abTestAPI, resultsAPI } from '@/utils/api';
import type { TimeSeriesResponse } from '@/types';

export interface ResultsTestSummary {
  test_id: string;
  test_name: string;
  status: string;
  primary_metric: string;
  variants: string[];
  total_users: number;
  completion_percentage: number;
  simulation_status?: string;
}


interface UseResultsDataState {
  tests: ResultsTestSummary[];
  selectedTestId: string;
  setSelectedTestId: (testId: string) => void;
  selectedTest: ResultsTestSummary | undefined;
  timeSeriesData: TimeSeriesResponse | null;
  loading: boolean;
  isSimulating: boolean;
  refreshCurrentTest: () => Promise<void>;
}

export const useResultsData = (): UseResultsDataState => {
  const [tests, setTests] = useState<ResultsTestSummary[]>([]);
  const [selectedTestId, setSelectedTestId] = useState<string>('');
  const [timeSeriesData, setTimeSeriesData] = useState<TimeSeriesResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [isSimulating, setIsSimulating] = useState(false);
  const pollingRef = useRef<number | null>(null);
  const initialLoadRef = useRef<string | null>(null);

  const loadTests = useCallback(async (): Promise<ResultsTestSummary[]> => {
    try {
      const response = await abTestAPI.getAllTests();
      const allTests: ResultsTestSummary[] = [
        ...(response.data.active_tests || []),
        ...(response.data.paused_tests || []),
        ...(response.data.completed_tests || []),
      ].map((t: any) => ({
        test_id: t.test_id,
        test_name: t.test_name,
        status: t.status,
        primary_metric: t.primary_metric,
        variants: t.variants,
        total_users: t.total_users,
        completion_percentage: t.completion_percentage,
        simulation_status: t.simulation_status,
      }));

      setTests((prevTests) => {
        if (prevTests.length !== allTests.length) return allTests;

        const nextById = new Map(allTests.map((test) => [test.test_id, test]));
        const hasDiff = prevTests.some((test) => {
          const next = nextById.get(test.test_id);
          return !next
            || next.status !== test.status
            || next.simulation_status !== test.simulation_status
            || next.total_users !== test.total_users
            || next.completion_percentage !== test.completion_percentage;
        });

        return hasDiff ? allTests : prevTests;
      });
      return allTests;
    } catch (error) {
      console.error('Ошибка загрузки тестов:', error);
      return [];
    }
  }, []);

  const loadTimeSeriesData = useCallback(async (testId: string, showLoading: boolean = true) => {
    if (showLoading) {
      setLoading(true);
    }

    try {
      const timeSeriesResponse = await resultsAPI.getTimeSeriesData(testId);

      setTimeSeriesData(timeSeriesResponse.data);
    } catch (error) {
      console.error('Ошибка загрузки результатов:', error);
      setTimeSeriesData(null);
    } finally {
      if (showLoading) {
        setLoading(false);
      }
    }
  }, []);

  const refreshCurrentTest = useCallback(async () => {
    if (!selectedTestId) return;
    await loadTimeSeriesData(selectedTestId, false);
  }, [loadTimeSeriesData, selectedTestId]);

  useEffect(() => {
    let mounted = true;

    const bootstrap = async () => {
      const allTests = await loadTests();
      if (!mounted) return;

      if (!selectedTestId && allTests.length > 0) {
        const completedTest = allTests.find((t) => t.status === 'completed');
        const fallback = completedTest ?? allTests[0];
        setSelectedTestId(fallback.test_id);
      }
    };

    bootstrap();

    return () => {
      mounted = false;
      if (pollingRef.current) {
        window.clearInterval(pollingRef.current);
      }
    };
  }, [loadTests, selectedTestId]);

  useEffect(() => {
    const currentlySimulating = tests.some((t) => t.simulation_status === 'running');
    setIsSimulating(currentlySimulating);
  }, [tests]);

  useEffect(() => {
    const selectedTest = tests.find((t) => t.test_id === selectedTestId);
    const shouldPollTests = isSimulating || selectedTest?.status === 'active' || selectedTest?.status === 'paused';

    if (!shouldPollTests) {
      return;
    }

    const testsInterval = window.setInterval(() => {
      loadTests();
    }, 5000);

    return () => {
      window.clearInterval(testsInterval);
    };
  }, [isSimulating, loadTests, selectedTestId, tests]);

  useEffect(() => {
    if (pollingRef.current) {
      window.clearInterval(pollingRef.current);
    }

    if (!selectedTestId) {
      return;
    }

    const selectedTest = tests.find((t) => t.test_id === selectedTestId);
    const selectedTestIsRunning =
      selectedTest?.simulation_status === 'running' ||
      selectedTest?.status === 'active' ||
      selectedTest?.status === 'paused';

    if (initialLoadRef.current !== selectedTestId) {
      loadTimeSeriesData(selectedTestId, true);
      initialLoadRef.current = selectedTestId;
    }

    if (!selectedTestIsRunning) {
      return;
    }

    const pollIntervalMs = selectedTest?.simulation_status === 'running' ? 3000 : 7000;

    pollingRef.current = window.setInterval(() => {
      loadTimeSeriesData(selectedTestId, false);
    }, pollIntervalMs);

    return () => {
      if (pollingRef.current) {
        window.clearInterval(pollingRef.current);
      }
    };
  }, [loadTimeSeriesData, selectedTestId, tests]);

  const selectedTest = useMemo(
    () => tests.find((t) => t.test_id === selectedTestId),
    [selectedTestId, tests],
  );

  return {
    tests,
    selectedTestId,
    setSelectedTestId,
    selectedTest,
    timeSeriesData,
    loading,
    isSimulating,
    refreshCurrentTest,
  };
};
